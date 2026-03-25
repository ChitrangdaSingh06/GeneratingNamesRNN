import torch
import LoadData

max_length = 20

def sample(rnn,category,start_letter="A"):
    with torch.no_grad():
        category_tensor = LoadData.categoryTensor(category)
        input  = LoadData.inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
             output, hidden = rnn(category_tensor, input[0], hidden)
             topv, topi = output.topk(1)
             if topi == LoadData.n_letters - 1:
                break
             else:
                letter = LoadData.all_letters[topi]
                output_name += letter
             input = LoadData.inputTensor(letter)
        
        return output_name

# Get multiple samples from one category and multiple starting letters
def samples(rnn,category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(rnn,category, start_letter))