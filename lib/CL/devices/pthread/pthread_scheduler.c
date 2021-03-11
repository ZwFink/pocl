/* OpenCL native pthreaded device implementation.

   Copyright (c) 2011-2019 pocl developers

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#define _GNU_SOURCE

#ifdef __linux__
#include <sched.h>
#endif

#include <pthread.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>
#include <sys/time.h>

#include "pocl-pthread_scheduler.h"
#include "pocl_cl.h"
#include "pocl-pthread.h"
#include "pocl-pthread_utils.h"
#include "utlist.h"
#include "pocl_util.h"
#include "common.h"
#include "pocl_mem_management.h"
void sig_func(int sig)
{
 write(1, "Caught signal 11\n", 17);
 signal(SIGINT,sig_func);
}

static void* pocl_pthread_driver_thread (void *p);
long get_elapsed(struct timeval *start, struct timeval *end)
{
  long sec_to_us = (long) (end->tv_sec - start->tv_sec) * 1000000L;
  long us_elapsed = (long) end->tv_usec - start->tv_usec;
  return sec_to_us + us_elapsed;
}

struct pool_thread_data
{
  pthread_t thread __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

  unsigned long executed_commands;
  /* per-CU (= per-thread) local memory */
  void *local_mem;
  unsigned current_ftz;
  unsigned num_threads;
  /* index of this particular thread
   * [0, num_threads-1]
   * used for deciding whether a particular thread should run
   * commands scheduled on a subdevice. */
  unsigned index;
  /* printf buffer*/
  void *printf_buffer;
  long total_wait_time_wq;
  long total_wait_time_kq;
  long total_exec_time;
  int num_waited_wq;
  int num_waited_kq;
  int num_executed;

  int num_cond_wait;
  long cond_wait_time;

  long first_wait;
  pthread_barrier_t *reset_barrier;
  int *do_reset;

  struct timeval time_thread_begin;
  struct timeval time_thread_end;
} __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

typedef struct scheduler_data_
{
  unsigned num_threads;
  unsigned printf_buf_size;

  struct pool_thread_data *thread_pool;
  size_t local_mem_size;

  _cl_command_node *work_queue
      __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
  kernel_run_command *kernel_queue;

  pthread_cond_t wake_pool __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
  POCL_FAST_LOCK_T wq_lock_fast __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

  int thread_pool_shutdown_requested;
} scheduler_data __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

static scheduler_data scheduler;

void
pthread_scheduler_init (cl_device_id device)
{
  unsigned i;
  size_t num_worker_threads = device->max_compute_units;
  pthread_barrier_t *reset_barrier = malloc(sizeof(pthread_barrier_t));
  int *do_reset = malloc(sizeof(int));
  *do_reset = 0;
  // all the worker threads and the master thread will wait for this barrier
  pthread_barrier_init(reset_barrier, NULL, num_worker_threads+1);
  printf("Using %zu threads!\n", num_worker_threads);
  POCL_FAST_INIT (scheduler.wq_lock_fast);

  pthread_cond_init (&(scheduler.wake_pool), NULL);

  scheduler.thread_pool = pocl_aligned_malloc (
      HOST_CPU_CACHELINE_SIZE,
      num_worker_threads * sizeof (struct pool_thread_data));
  memset (scheduler.thread_pool, 0,
          num_worker_threads * sizeof (struct pool_thread_data));

  scheduler.num_threads = num_worker_threads;
  assert (num_worker_threads > 0);
  scheduler.printf_buf_size = device->printf_buffer_size;
  assert (device->printf_buffer_size > 0);

  /* safety margin - aligning pointers later (in kernel arg setup)
   * may require more local memory than actual local mem size.
   * TODO fix this */
  scheduler.local_mem_size = device->local_mem_size + device->max_parameter_size * MAX_EXTENDED_ALIGNMENT;

  for (i = 0; i < num_worker_threads; ++i)
    {

      scheduler.thread_pool[i].index = i;
      scheduler.thread_pool[i].reset_barrier = reset_barrier;
      scheduler.thread_pool[i].do_reset = do_reset;
      scheduler.thread_pool[i].total_wait_time_wq = 0;
      scheduler.thread_pool[i].total_wait_time_kq = 0;
      scheduler.thread_pool[i].total_exec_time = 0;
      scheduler.thread_pool[i].cond_wait_time = 0;

      scheduler.thread_pool[i].num_cond_wait = 0;
      scheduler.thread_pool[i].num_waited_wq = 0;
      scheduler.thread_pool[i].num_waited_kq = 0;
      scheduler.thread_pool[i].num_executed = 0;

      pthread_create (&scheduler.thread_pool[i].thread, NULL,
                      pocl_pthread_driver_thread,
                      (void*)&scheduler.thread_pool[i]);
    }

}

void
pthread_scheduler_uninit ()
{
  unsigned i;

  POCL_FAST_LOCK (scheduler.wq_lock_fast);
  scheduler.thread_pool_shutdown_requested = 1;
  pthread_cond_broadcast (&scheduler.wake_pool);
  POCL_FAST_UNLOCK (scheduler.wq_lock_fast);

  for (i = 0; i < scheduler.num_threads; ++i)
    {
      pthread_join (scheduler.thread_pool[i].thread, NULL);
    }

  int num_waited_total = 0;
  int total_waited_wq = 0;
  int total_waited_kq = 0;
  int total_cond_wait = 0;
  int total_executed = 0;

  long time_waited_total = 0;
  long total_time_wq = 0;
  long total_time_kq = 0;
  long total_time_alive = 0;
  long total_time_cond_wait = 0;
  long total_exec_time = 0;

  printf("TIMING_ANALYSIS_INDIV: Thread Id,Total Num Wait,Total wait time, Num wait Queue, Num Kernel Queue,Time Wait queue, Time Kernel Queue,Time Alive,Num Cond Wait, Cond Wait Time,Num Exec Calls,Time Exec\n");
  for(i = 0; i < scheduler.num_threads; ++i)
    {
      struct pool_thread_data *td = &scheduler.thread_pool[i];
      long time_alive = get_elapsed(&td->time_thread_begin, &td->time_thread_end);

      int num_waited_wq = td->num_waited_wq;
      int num_waited_kq = td->num_waited_kq;
      int num_waited_cond = td->num_cond_wait;
      int num_executed = td->num_executed;


      long time_waited_wq = td->total_wait_time_wq;
      long time_waited_kq = td->total_wait_time_kq;
      long time_waited_cond = td->cond_wait_time;
      long exec_time = td->total_exec_time;

      printf("TIMING_ANALYSIS_INDIV:%d,%d%ld,%d,%ld,%d,%ld,%ld,%d,%ld,%d,%ld\n",
             td->index,
             num_waited_wq + num_waited_kq,
             time_waited_kq + time_waited_wq,
             num_waited_wq, time_waited_wq,
             num_waited_kq, time_waited_kq,
             time_alive,
             num_waited_cond, time_waited_cond,
             num_executed, exec_time
             );

      num_waited_total += num_waited_wq + num_waited_kq;
      total_waited_wq += num_waited_wq;
      total_waited_kq += num_waited_kq;
      total_cond_wait += num_waited_cond;
      time_waited_total += time_waited_wq + time_waited_kq;
      total_time_wq += time_waited_wq;
      total_time_kq += time_waited_kq;
      total_time_alive += time_alive;
      total_time_cond_wait += time_waited_cond;
      total_executed += num_executed;
      total_exec_time += exec_time;
    }
  printf("TIMING_ANALYSIS_AGG: Total Num Wait, Num Wait Work Queue, Num Wait Kernel Queue,Total Wait Time, Work Queue Time, "
         "Kernel Queue Time, Avg Wait Time, Total Time All Threads,Num Wait Cond,Time wait cond,Total Exec Calls, Total Exec Time\n");
  printf("TIMING_ANALYSIS_AGG: %d,%d,%d,%ld,%ld,%ld,%f,%ld,%d,%ld,%d,%ld\n",
         num_waited_total, total_waited_wq, total_waited_kq, time_waited_total, total_time_wq, total_time_kq,
         ((double)time_waited_total)/(double)num_waited_total, total_time_alive, total_cond_wait, total_time_cond_wait, total_executed, total_exec_time
         );

  /* pocl_aligned_free (scheduler.thread_pool); */
  /* POCL_FAST_DESTROY (scheduler.wq_lock_fast); */
  /* pthread_cond_destroy (&scheduler.wake_pool); */

  scheduler.thread_pool_shutdown_requested = 0;
}

/* push_command and push_kernel MUST use broadcast and wake up all threads,
   because commands can be for subdevices (= not all threads) */
void pthread_scheduler_push_command (_cl_command_node *cmd)
{
  POCL_FAST_LOCK (scheduler.wq_lock_fast);
  DL_APPEND (scheduler.work_queue, cmd);
  pthread_cond_broadcast (&scheduler.wake_pool);
  POCL_FAST_UNLOCK (scheduler.wq_lock_fast);
}

static void
pthread_scheduler_push_kernel (kernel_run_command *run_cmd)
{
  struct timeval wait_start;
  struct timeval wait_end;

  /* execute kernel if available */
  // work queue lock fast
  POCL_FAST_LOCK (scheduler.wq_lock_fast);
  DL_APPEND (scheduler.kernel_queue, run_cmd);
  pthread_cond_broadcast (&scheduler.wake_pool);
  POCL_FAST_UNLOCK (scheduler.wq_lock_fast);
}

/* if subd is not a subdevice, returns 1
 * if subd is subdevice, takes a look at the subdevice CUs
 * and if they match the current driver thread, returns 1
 * otherwise returns 0 */
static int
shall_we_run_this (thread_data *td, cl_device_id subd)
{

  if (subd && subd->parent_device)
    {
      if (!((td->index >= subd->core_start)
            && (td->index < (subd->core_start + subd->core_count))))
        {
          return 0;
        }
    }
  return 1;
}

/* Maximum and minimum chunk sizes for get_wg_index_range().
 * Each pthread driver's thread fetches work from a kernel's WG pool in
 * chunks, this determines the limits (scaled up by # of threads). */
#define POCL_PTHREAD_MAX_WGS 256
#define POCL_PTHREAD_MIN_WGS 32

static int
get_wg_index_range (kernel_run_command *k, unsigned *start_index,
                    unsigned *end_index, int *last_wgs, unsigned num_threads,
                    thread_data *td)
{
  const unsigned scaled_max_wgs = POCL_PTHREAD_MAX_WGS * num_threads;
  const unsigned scaled_min_wgs = POCL_PTHREAD_MIN_WGS * num_threads;
  struct timeval wait_start;
  struct timeval wait_end;

  /* execute kernel if available */
  // work queue lock fast


  unsigned limit;
  unsigned max_wgs;
  gettimeofday(&wait_start, NULL);
  POCL_FAST_LOCK (k->lock);
  gettimeofday(&wait_end, NULL);
  td->total_wait_time_kq += get_elapsed(&wait_start, &wait_end);
  td->num_waited_kq++;


  if (k->remaining_wgs == 0)
    {
      POCL_FAST_UNLOCK (k->lock);
      return 0;
    }

  /* If the work is comprised of huge number of WGs of small WIs,
   * then get_wg_index_range() becomes a problem on manycore CPUs
   * because lock contention on k->lock.
   *
   * If we have enough workgroups, scale up the requests linearly by
   * num_threads, otherwise fallback to smaller workgroups.
   */
  if (k->remaining_wgs <= (scaled_max_wgs * num_threads))
    limit = scaled_min_wgs;
  else
    limit = scaled_max_wgs;

  // divide two integers rounding up, i.e. ceil(k->remaining_wgs/num_threads)
  const unsigned wgs_per_thread = (1 + (k->remaining_wgs - 1) / num_threads);
  max_wgs = min (limit, wgs_per_thread);
  max_wgs = min (max_wgs, k->remaining_wgs);
  assert (max_wgs > 0);

  *start_index = k->wgs_dealt;
  *end_index = k->wgs_dealt + max_wgs-1;
  k->remaining_wgs -= max_wgs;
  k->wgs_dealt += max_wgs;
  if (k->remaining_wgs == 0)
    *last_wgs = 1;
  POCL_FAST_UNLOCK (k->lock);

  return 1;
}

inline static void translate_wg_index_to_3d_index (kernel_run_command *k,
                                                   unsigned index,
                                                   size_t *index_3d,
                                                   unsigned xy_slice,
                                                   unsigned row_size)
{
  index_3d[2] = index / xy_slice;
  index_3d[1] = (index % xy_slice) / row_size;
  index_3d[0] = (index % xy_slice) % row_size;
}

static int
work_group_scheduler (kernel_run_command *k,
                      struct pool_thread_data *thread_data)
{
  pocl_kernel_metadata_t *meta = k->kernel->meta;

  void *arguments[meta->num_args + meta->num_locals + 1];
  void *arguments2[meta->num_args + meta->num_locals + 1];
  struct pocl_context pc;
  unsigned i;
  unsigned start_index;
  unsigned end_index;
  int last_wgs = 0;

  if (!get_wg_index_range (k, &start_index, &end_index, &last_wgs,
                           thread_data->num_threads, thread_data))
    return 0;

  assert (end_index >= start_index);

  setup_kernel_arg_array_with_locals (
      (void **)&arguments, (void **)&arguments2, k, thread_data->local_mem,
      scheduler.local_mem_size);
  memcpy (&pc, &k->pc, sizeof (struct pocl_context));

  // capacity and position already set up
  pc.printf_buffer = thread_data->printf_buffer;
  uint32_t position = 0;
  pc.printf_buffer_position = &position;
  assert (pc.printf_buffer != NULL);
  assert (pc.printf_buffer_capacity > 0);
  assert (pc.printf_buffer_position != NULL);

  /* Flush to zero is only set once at start of kernel (because FTZ is
   * a compilation option), but we need to reset rounding mode after every
   * iteration (since it can be changed during kernel execution). */
  unsigned flush = k->kernel->program->flush_denorms;
  if (thread_data->current_ftz != flush)
    {
      pocl_set_ftz (flush);
      thread_data->current_ftz = flush;
    }

  unsigned slice_size = k->pc.num_groups[0] * k->pc.num_groups[1];
  unsigned row_size = k->pc.num_groups[0];

  struct timeval wait_start;
  struct timeval wait_end;

  do
    {
      if (last_wgs)
        {
          gettimeofday(&wait_start, NULL);
          POCL_FAST_LOCK (scheduler.wq_lock_fast);
          gettimeofday(&wait_end, NULL);
          DL_DELETE (scheduler.kernel_queue, k);
          thread_data->total_wait_time_wq += get_elapsed(&wait_start, &wait_end);
          thread_data->num_waited_wq++;
          POCL_FAST_UNLOCK (scheduler.wq_lock_fast);
        }

      for (i = start_index; i <= end_index; ++i)
        {
          size_t gids[3];
          translate_wg_index_to_3d_index (k, i, gids,
                                          slice_size, row_size);

#ifdef DEBUG_MT
          printf("### exec_wg: gid_x %zu, gid_y %zu, gid_z %zu\n",
                 gids[0], gids[1], gids[2]);
#endif
          pocl_set_default_rm ();

          gettimeofday(&wait_start, NULL);
          k->workgroup ((uint8_t*)arguments, (uint8_t*)&pc,
                        gids[0], gids[1], gids[2]);
          gettimeofday(&wait_end, NULL);
          thread_data->total_exec_time += get_elapsed(&wait_start, &wait_end);
          thread_data->num_executed++;
        }
    }
  while (get_wg_index_range (k, &start_index, &end_index, &last_wgs,
                             thread_data->num_threads, thread_data));

  if (position > 0)
    {
      write (STDOUT_FILENO, pc.printf_buffer, position);
    }

  free_kernel_arg_array_with_locals ((void **)&arguments, (void **)&arguments2,
                                     k);

  return 1;
}

static void
finalize_kernel_command (struct pool_thread_data *thread_data,
                         kernel_run_command *k)
{
#ifdef DEBUG_MT
  printf("### kernel %s finished\n", k->cmd->command.run.kernel->name);
#endif

  free_kernel_arg_array (k);

  pocl_release_dlhandle_cache (k->cmd);

  pocl_ndrange_node_cleanup (k->cmd);

  POCL_UPDATE_EVENT_COMPLETE_MSG (k->cmd->event, "NDRange Kernel        ");

  pocl_mem_manager_free_command (k->cmd);
  POCL_FAST_DESTROY (k->lock);
  free_kernel_run_command (k);
}

static void
pocl_pthread_prepare_kernel (void *data, _cl_command_node *cmd)
{
  kernel_run_command *run_cmd;
  cl_kernel kernel = cmd->command.run.kernel;
  struct pocl_context *pc = &cmd->command.run.pc;

  pocl_check_kernel_dlhandle_cache (cmd, 1, 1);

  size_t num_groups = pc->num_groups[0] * pc->num_groups[1] * pc->num_groups[2];

  run_cmd = new_kernel_run_command ();
  run_cmd->data = data;
  run_cmd->kernel = kernel;
  run_cmd->device = cmd->device;
  run_cmd->pc = *pc;
  run_cmd->cmd = cmd;
  run_cmd->pc.printf_buffer = NULL;
  run_cmd->pc.printf_buffer_capacity = scheduler.printf_buf_size;
  run_cmd->pc.printf_buffer_position = NULL;
  run_cmd->remaining_wgs = num_groups;
  run_cmd->wgs_dealt = 0;
  run_cmd->workgroup = cmd->command.run.wg;
  run_cmd->kernel_args = cmd->command.run.arguments;
  run_cmd->next = NULL;
  run_cmd->ref_count = 0;
  POCL_FAST_INIT (run_cmd->lock);

  setup_kernel_arg_array (run_cmd);

  pocl_update_event_running (cmd->event);

  pthread_scheduler_push_kernel (run_cmd);
}

/*
  These two check the entire kernel/cmd queue. This is necessary
  because of commands for subdevices. The old code only checked
  the head of each queue; this can lead to a deadlock:

  two driver threads, each assigned two subdevices A, B, one
  driver queue C

  cmd A1 for A arrives in C, A starts processing
  cmd B1 for B arrives in C, B starts processing
  cmds A2, A3, B2 are pushed to C
  B finishes processing B1, checks queue head, A2 isn't for it, goes to sleep
  A finishes processing A1, processes A2 + A3 but ignores B2, it's not for it
  application calls clFinish to wait for queue

  ...now B is sleeping and not possible to wake up
  (since no new commands can arrive) and there's a B2 command
  which will never be processed.

  it's possible to workaround but it's cleaner to just check the whole queue.
 */

static _cl_command_node *
check_cmd_queue_for_device (thread_data *td)
{
  _cl_command_node *cmd;
  DL_FOREACH (scheduler.work_queue, cmd)
  {
    cl_device_id subd = cmd->device;
    if (shall_we_run_this (td, subd))
      {
        DL_DELETE (scheduler.work_queue, cmd);
        return cmd;
      }
  }

  return NULL;
}

static kernel_run_command *
check_kernel_queue_for_device (thread_data *td)
{
  kernel_run_command *cmd;
  DL_FOREACH (scheduler.kernel_queue, cmd)
  {
    cl_device_id subd = cmd->device;
    if (shall_we_run_this (td, subd))
      return cmd;
  }

  return NULL;
}

static int
pthread_scheduler_get_work (thread_data *td)
{
  _cl_command_node *cmd;
  kernel_run_command *run_cmd;
  struct timeval wait_start;
  struct timeval wait_end;

  /* execute kernel if available */
  // work queue lock fast
  gettimeofday(&wait_start, NULL);
  POCL_FAST_LOCK (scheduler.wq_lock_fast);
  gettimeofday(&wait_end, NULL);
  td->total_wait_time_wq += get_elapsed(&wait_start, &wait_end);
  td->num_waited_wq++;

  int do_reset = 0;
  int do_exit = 0;

RETRY:
  // should we shutdown this??
  do_exit = scheduler.thread_pool_shutdown_requested;
  do_reset = *td->do_reset;

  // is this thread data associated with a kernel run command?
  // is there a device available to run this kernel?
  run_cmd = check_kernel_queue_for_device (td);
  // if this is a kernel to execute, we should execute it
  /* execute kernel if available */
  if (run_cmd)
    {
      ++run_cmd->ref_count;


      POCL_FAST_UNLOCK (scheduler.wq_lock_fast);


      // if so, we wan tto go ahead and run it
      work_group_scheduler (run_cmd, td);

      gettimeofday(&wait_start, NULL);

      POCL_FAST_LOCK (scheduler.wq_lock_fast);
      gettimeofday(&wait_end, NULL);
      td->total_wait_time_wq += get_elapsed(&wait_start, &wait_end);
      td->num_waited_wq++;


      // are we the last thread to have a reference to the kernel?
      if ((--run_cmd->ref_count) == 0)
        {
          POCL_FAST_UNLOCK (scheduler.wq_lock_fast);
          finalize_kernel_command (td, run_cmd);

          gettimeofday(&wait_start, NULL);
          POCL_FAST_LOCK (scheduler.wq_lock_fast);
          gettimeofday(&wait_end, NULL);
          td->total_wait_time_wq += get_elapsed(&wait_start, &wait_end);
          td->num_waited_wq++;
        }
    }

  /* execute a command if available */
  cmd = check_cmd_queue_for_device (td);
  if (cmd)
    {
      POCL_FAST_UNLOCK (scheduler.wq_lock_fast);

      assert (pocl_command_is_ready (cmd->event));

      if (cmd->type == CL_COMMAND_NDRANGE_KERNEL)
        {
          pocl_pthread_prepare_kernel (cmd->device->data, cmd);
        }
      else
        {
          pocl_exec_command (cmd);
        }

      gettimeofday(&wait_start, NULL);

      POCL_FAST_LOCK (scheduler.wq_lock_fast);
      gettimeofday(&wait_end, NULL);
      td->total_wait_time_wq += get_elapsed(&wait_start, &wait_end);
      td->num_waited_wq++;
      ++td->executed_commands;
    }

  /* if neither a command nor a kernel was available, sleep */
  if ((cmd == NULL) && (run_cmd == NULL) && (do_exit == 0) && do_reset == 0)
    {
      gettimeofday(&wait_start, NULL);
      pthread_cond_wait (&scheduler.wake_pool, &scheduler.wq_lock_fast);
      gettimeofday(&wait_end, NULL);
      td->num_cond_wait++;
      td->cond_wait_time += get_elapsed(&wait_start, &wait_end);
      if(td->num_cond_wait == 1)
        {
          td->first_wait = td->cond_wait_time;
          printf("First Wait: %ld\n", td->cond_wait_time);
        }
      if(td->num_cond_wait == 2)
        printf("Second Wait: %ld\n", td->cond_wait_time-td->first_wait);

      goto RETRY;
    }

  POCL_FAST_UNLOCK (scheduler.wq_lock_fast);

  return do_exit;
}

// to be called from python
void reset_thread_data()
{
  *scheduler.thread_pool[0].do_reset = 1;
  pthread_cond_broadcast (&scheduler.wake_pool);
  pthread_barrier_wait(scheduler.thread_pool[0].reset_barrier);
}

void reset_data(struct pool_thread_data *td)
{
      td->total_wait_time_wq = 0;
      td->total_wait_time_kq = 0;
      td->total_exec_time = 0;
      td->cond_wait_time = 0;

      td->num_cond_wait = 0;
      td->num_waited_wq = 0;
      td->num_waited_kq = 0;
      td->num_executed = 0;
}

static
void*
pocl_pthread_driver_thread (void *p)
{
  struct pool_thread_data *td = (struct pool_thread_data*)p;
  int do_exit = 0;
  assert (td);
  signal(SIGINT,sig_func); // Register signal handler before going multithread
  /* some random value, doesn't matter as long as it's not a valid bool - to
   * force a first FTZ setup */
  td->current_ftz = 213;
  td->num_threads = scheduler.num_threads;
  td->printf_buffer = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT,
                                           scheduler.printf_buf_size);
  assert (td->printf_buffer != NULL);

  assert (scheduler.local_mem_size > 0);
  td->local_mem = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT,
                                       scheduler.local_mem_size);
  assert (td->local_mem);
#ifdef __linux__
  if (1)
    {
      cpu_set_t set;
      CPU_ZERO (&set);
      CPU_SET (td->index, &set);
      pthread_setaffinity_np (td->thread, sizeof (cpu_set_t), &set);
    }
#endif
  int has_done = 0;

  while (1)
    {

      if(!has_done)
        {
          has_done = 1;
          gettimeofday(&td->time_thread_begin, NULL);
        }
      do_exit = pthread_scheduler_get_work (td);
      if (do_exit)
        {
          gettimeofday(&td->time_thread_end, NULL);
          pocl_aligned_free (td->printf_buffer);
          pocl_aligned_free (td->local_mem);
          pthread_exit (NULL);
        }
      int do_reset = *td->do_reset;
      if(do_reset)
        {
          reset_data(td);
          has_done = 0;
          pthread_barrier_wait(td->reset_barrier);
          *(td->do_reset) = 0;
        }
    }
}
