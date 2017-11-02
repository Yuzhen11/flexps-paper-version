Evaulate the stage transition cost by measuring the time between last stage finish time
and the current stage (empty) finsih time.

The time includes: stage scheduler do scheduling and all threads finish initializing the consistency controller.

Measure the time using the time between [Task 0 epoch 0 finished] and [Task 0 epoch 1 finished].

It takes 0.014s to finish on 20 processes each with 10 worker threads.

