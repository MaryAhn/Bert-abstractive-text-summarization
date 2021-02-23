path_open_original = '/Users/angeonhui/Bert-abstractive-text-summarization/data/checkpoint/vocab_snu.txt'
path_open_made = '/Users/angeonhui/Bert-abstractive-text-summarization/data/checkpoint/vocab_made.txt'
path_save = '/Users/angeonhui/Bert-abstractive-text-summarization/data/dataset/for_vocab/revised_vocab_snu_fromFirst.txt'

original_file = open(path_open_original, 'r')
made_file = open(path_open_made, 'r')
new_file = open(path_save, 'w')

original_list = original_file.readlines()
made_list = made_file.readlines()
new_list = []

print(len(original_list), len(made_list))

for i in range(len(made_list)):
    # print(original_list[i], made_list[i])
    original_list[i] = made_list[i]

new_list = original_list

for i in range(len(new_list)):
    new_file.write(new_list[i])

new_file.close()



