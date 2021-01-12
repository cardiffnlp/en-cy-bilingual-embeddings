from googletrans import Translator

# perform, perform, serious (antibon), lifting (liqu1func0)

sents = """the error message reads : this file does not have a program associated with it for performing this action .
scientists in italy discovered " mirror neurons " that respond when we see someone else perform an action - or even when we hear an action described - as if we ourselves were performing the action .
those injured were sent to nearby hospital , and one passenger was in serious condition .
last year , many who opposed lifting the ban on gays in the military gave lip service to the american ideal that employment opportunities should be based on skill and performance ."""

translator = Translator()

langs = ['es', 'fr']

for sent in sents.split('\n'):

	for l in langs:

		print('Original: ',sent)
		trans = translator.translate(sent, dest=l)
		back = translator.translate(trans.text, dest='en')
		print(f'Para ({l}): ',back.text)
		print('\n---\n')
		
