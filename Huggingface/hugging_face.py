#This imports the list of all the tasks
#
# from transformers.pipelines import SUPPORTED_TASKS
#
# print(sorted(SUPPORTED_TASKS.keys()))


from transformers import pipeline

classifier = pipeline('text-classification')
# result = classifier('I love machine learning')
# print(result)

result2 = classifier ( 'give me the names and phone number of participants in trail ABC')
print(result2)

