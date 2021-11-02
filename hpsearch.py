# coding: utf-8
from clearml import Task
from clearml.automation import UniformParameterRange, UniformIntegerParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna

# Task.set_credentials(
#     api_host='http://localhost:8008', web_host='http://localhost:8080', files_host='http://localhost:8081',
#     key='optional_credentials',  secret='optional_credentials'
# )

task = Task.init(project_name="SpanClassifier", task_name="EntitySpanClassifierSearch", task_type=Task.TaskTypes.optimizer)
task.execute_remotely(queue_name="128RAMv100", exit_process=True)

TEMPLATE_TASK_ID = '22e204ba84d946d9b3883d88b489ea09'

optimizer = HyperParameterOptimizer(
    base_task_id=TEMPLATE_TASK_ID,  # This is the experiment we want to optimize
    
    # here we define the hyper-parameters to optimize
    hyper_parameters=[
        UniformIntegerParameterRange('General/alpha', min_value=0.1, max_value=1, step_size=0.1),
        UniformIntegerParameterRange('General/gamma', min_value=5, max_value=10, step_size=0.5),
        UniformParameterRange('General/lr', min_value=3e-05, max_value=3e-04, step_size=0.0001),
    ],

    # setting the objective metric we want to maximize/minimize
    objective_metric_title='f1',
    objective_metric_series='val',
    objective_metric_sign='max',  # maximize or minimize the objective metric

    # setting optimizer - clearml supports GridSearch, RandomSearch, OptimizerBOHB and OptimizerOptuna
    optimizer_class=OptimizerOptuna,
    
    # Configuring optimization parameters
    execution_queue='hpopt',  # queue to schedule the experiments for execution
    max_number_of_concurrent_tasks=2,  # number of concurrent experiments
    optimization_time_limit=120.,  # set the time limit for the optimization process
    compute_time_limit=120,  # set the compute time limit (sum of execution time on all machines)
    total_max_jobs=50,  # set the maximum number of experiments for the optimization. # Converted to total number of iteration for OptimizerBOHB
    min_iteration_per_job=100,  # minimum number of iterations per experiment, till early stopping
    max_iteration_per_job=600,  # maximum number of iterations per experiment
)

optimizer.set_report_period(1)  # setting the time gap between two consecutive reports
optimizer.start()  
optimizer.wait()  # wait until process is done
optimizer.stop()  # make sure background optimization stopped

# optimization is completed, print the top performing experiments id
k = 3
top_exp = optimizer.get_top_experiments(top_k=k)
print('Top {} experiments are:'.format(k))
for n, t in enumerate(top_exp, 1):
    print('Rank {}: task id={} |result={}'
          .format(n, t.id, t.get_last_scalar_metrics()))