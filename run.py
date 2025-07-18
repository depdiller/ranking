import autorag
from autorag.validator import Validator
from llama_index.llms.anthropic import Anthropic
from autorag.evaluator import Evaluator
from autorag.dashboard import run

autorag.generator_models['anthropic'] = Anthropic

# validator = Validator(qa_data_path='data/qa.parquet', corpus_data_path='data/corpus.parquet')
# validator.validate('config.yaml')

# evaluator = Evaluator(qa_data_path='data/qa.parquet', corpus_data_path='data/corpus.parquet',
#                       project_dir='rag-tools-test')
# evaluator.start_trial('config.yaml', skip_validation=True, full_ingest=False)

# run(trial_dir='rag-tools-test/2')

# from autorag.deploy import ApiRunner
# import nest_asyncio
#
# nest_asyncio.apply()
#
# runner = ApiRunner.from_yaml('config.yaml', project_dir='rag-tools-test')
# runner.run_api_server()

# from autorag.deploy import GradioRunner
#
# runner = GradioRunner.from_trial_folder('rag-tools-test/2')
# runner.run_web()


from autorag.deploy import Runner

runner = Runner.from_trial_folder('rag-tools-test/2')
runner.run('How to create instances?')