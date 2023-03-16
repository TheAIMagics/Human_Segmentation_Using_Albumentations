from src.human.pipeline.training_pipeline import TrainingPipeline
from src.human.pipeline.prediction_pipeline import SinglePrediction

#pipeline = TrainingPipeline()

#pipeline.run_pipeline()

pipe = SinglePrediction()
pipe.predict(filename='70.jpg')