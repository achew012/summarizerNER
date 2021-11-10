from clearml import Task, StorageManager, Dataset, PipelineController
import os, json, jsonlines


from clearml import Task, StorageManager, Dataset, PipelineController

# Creating the pipeline
pipe = PipelineController(project = "pipeline", name="span-guided-qa", version='2')
pipe.set_default_execution_queue("128RAMv100")

pipe.add_step(name='span_clf', base_task_id='1c2b005c2e7e406f96a61adb05b492c2', execution_queue="compute")
pipe.add_step(name='qa', base_task_id='17cc79f0dae0426d9354fe08d979980a', execution_queue="compute")

pipe.start()
# Wait until pipeline terminates
pipe.wait()
# cleanup everything
pipe.stop()  
print('pipeline completed')






