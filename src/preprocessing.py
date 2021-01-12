import tqdm
import gensim


def naive_tokenisation(data):
    data = data.lower()
    punctuation = ['"', '”' "'", "‘",'#','$','%','(',')','[',']','{','}','*','+',',','-','_', '/','\\',':',';','<','>','@','^','`','\xa0','¡','¢','£','¥','§','¨','«','¬','®','¯','°','±','²','³','´','µ','·','¹','»','¼','½','¾','¿','×','₤','₦','₪','€','ℂ','ℓ','ℝ','ℤ','⅔','⅛','↑','→','↔','⇌','⇒','⇔','∀','∅','∈','∏','∑','−','∖','∗','∘','√','∝','∞','∥','∦','∧','∨','∩','∪','∫','≃','≅','≈','≔','≘','≝','≠','≡','≤','≥','≪','⊃','⊆','⋅','⋕','⌀','⌊','⌋','─','█','△','☆','☉','♀','♂','♈','♉','♊','♋','♌','♍','♎','♏','♐','♑','♒','♓','♞','♥','♦','♩','♪','♫','♬','♭','⟨','⟩','⟺','ⱱ','ⲁ','ⲉ','ⲏ','ⲓ','ⲕ','ⲗ','ⲙ','ⲛ','ⲟ','ⲣ','ⲧ','ⲩ','、','・','ﬀ','ﬁ','ﬂ','ﬅ','ﬆ','（','）','，','；','�','💖','–', ',', 'ˌ', 'ɛ', 'ː', '|', '\u2060', '•', '\u200e', '\ufeff', '″', '′', 'ˊ', '\u200b', '₂', '₇', '₃', 'ʻ', '…', '\u200c', '\x92', '\x96', '~', '\u2009', '‒', '\u202a', '\u202c', '\u2002']
    for punct in tqdm.tqdm(punctuation):
        data = data.replace(punct, '')
    data = data.replace('!', '.').replace('?', '.')

    paragraphs = data.split('\n')
    sentences = []
    for p in tqdm.tqdm(paragraphs):
        if len(p) != 0:
            sentences_in_paragraph = p.split('. ')
            for sentence in sentences_in_paragraph:
                sentences.append([])
                for word in sentence.replace('.', '').split(' '):
                    if word not in ['', ' ']:
                        sentences[-1].append(word)

    tokenised_data = '\n'.join([' '.join(sentence) for sentence in sentences])
    return tokenised_data


def gensim_tokenisation(data):
    sentences = gensim.summarization.textcleaner.get_sentences(data)
    all_sentences = []
    for sentence in sentences:
        sentence_tokenized = gensim.utils.tokenize(sentence, deacc=False, lowercase=True)
        all_sentences.append([word for word in sentence_tokenized])

    tokenised_data = '\n'.join([' '.join(sentence) for sentence in all_sentences])
    return tokenised_data
