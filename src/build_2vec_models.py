# Imports
import gensim.models.word2vec
import gensim.models.doc2vec
import numpy.random
import tarfile
import tika.parser
import ujson as json

# LexNLP imports
from lexnlp.nlp.en.segments.sentences import get_sentence_list
from lexnlp.nlp.en.tokens import get_token_list, get_stem_list

# Setup input
TAR_FILE_PATH = "../data/opinions/all.tar"

# Proportion of each jurisdiction tar file to include in sample (selected randomly)
CORPUS_PROPORTION = 0.1

# Number of workers to use in gensim model building
NUM_WORKERS = 1

def process_text(text):
    stems = [s for s in get_stem_list(text, stopword=True, lowercase=True) if s.isalpha()]
    return stems

if __name__ == "__main__":
    # setup key storage
    sentences = []
    documents = []

    # open outer corpus file all.tar
    with tarfile.open(TAR_FILE_PATH, "r") as corpus_tar_file:
        corpus_member_list = corpus_tar_file.getmembers()
        corpus_num_members = len(corpus_member_list)

        # Iterate through all .tar.gz within
        for i, corpus_tar_member in enumerate(corpus_member_list):
            print((TAR_FILE_PATH, i, corpus_tar_member.name))

            # Open file object if tar.gz
            if not corpus_tar_member.name.lower().endswith(".tar.gz"):
                continue
            corpus_member_fileobj = corpus_tar_file.extractfile(corpus_tar_member.name)
            with tarfile.open(fileobj=corpus_member_fileobj, mode="r:gz") as court_tar_file:
                # get file list within jurisdiction
                court_member_list = court_tar_file.getmembers()
                court_num_members = len(court_member_list)
                court_sample_size = int(court_num_members * CORPUS_PROPORTION)
                if court_sample_size > 0:
                    court_sample_list = numpy.random.choice(court_member_list, court_sample_size)
                else:
                    continue

                # iterate through opinions
                for j, court_tar_member in enumerate(court_sample_list):
                    print((TAR_FILE_PATH, i, corpus_tar_member.name,
                           court_tar_member.name, j))

                    # load json data
                    court_member_file = court_tar_file.extractfile(court_tar_member.name)
                    court_json_data = json.load(court_member_file)

                    # select content source and get text content
                    try:
                        if len(court_json_data['plain_text']) > 0:
                            if "<p" in court_json_data['plain_text'].lower():
                                html_content = "<html>{0}</html>".format(court_json_data['plain_text'])
                                tika_response = tika.parser.from_buffer(html_content)
                                text_content = tika_response['content']
                            else:
                                text_content = court_json_data['plain_text']
                        elif len(court_json_data['html_with_citations']) > 0:
                            html_content = "<html>{0}</html>".format(court_json_data['html_with_citations'])
                            tika_response = tika.parser.from_buffer(html_content)
                            text_content = tika_response['content']
                        else:
                            text_content = ''
                    except Exception as e:
                        print(("error in content extraction", e))
                        continue

                    # skip if empty
                    if text_content is None:
                        continue
                    if len(text_content.strip()) == 0:
                        continue

                    try:
                        # build word2vec sentence list and doc2vec content simultaneously
                        doc_stems = []
                        for sentence in get_sentence_list(text_content):
                            sentence_stems = [s for s in get_stem_list(sentence, stopword=True, lowercase=True) if s.isalpha()]
                            doc_stems.extend(sentence_stems)
                            sentences.append(sentence_stems)
                        documents.append(gensim.models.doc2vec.TaggedDocument(doc_stems, ["{0}".format(court_tar_member.name)]))
                    except Exception as e:
                        print(e)
                        
    # word2vec models
    min_count = 10
    w2v_size_list = [100, 200]
    w2v_window_list = [5, 10, 20]
    for size in w2v_size_list:
        for window in w2v_window_list:
            w2v_model_cbow = gensim.models.word2vec.Word2Vec(sentences, size=size, window=window, min_count=min_count, workers=NUM_WORKERS)
            w2v_model_cbow.save("../data/models/w2v_cbow_all_size{0}_window{1}".format(size, window))
            
    # doc2vec models
    min_count = 10
    d2v_size_list = [100, 200]
    d2v_window_list = [5, 10, 20]
    for size in d2v_size_list:
        for window in d2v_window_list:
            d2v_model = gensim.models.doc2vec.Doc2Vec(documents, vector_size=size, window=window, min_count=min_count, workers=NUM_WORKERS)
            d2v_model.save("../data/models/d2v_all_size{0}_window{1}".format(size, window))

    
