########################################################################
########################################################################
##                                                                    ##
##                      ORIGINAL _ DO NOT PUBLISH                     ##
##                                                                    ##
########################################################################
########################################################################

# import torch as tr
import torch
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
import loader as ld


batch_size = 32
output_size = 2
hidden_size = 64        # to experiment with

run_recurrent = False    # else run Token-wise MLP
use_RNN = False          # otherwise GRU
atten_size = 0          # atten > 0 means using restricted self atten

reload_model = False
num_epochs = 7
learning_rate = 0.001
test_interval = 50

# Loading sataset, use toy = True for obtaining a smaller dataset

train_dataset, test_dataset, num_words, input_size = ld.get_data_set(batch_size)

# Special matrix multipication layer (like torch.Linear but can operate on arbitrary sized
# tensors and considers its last two indices as the matrix.)

class MatMul(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias = True):
        super(MatMul, self).__init__()
        self.matrix = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(in_channels,out_channels)), requires_grad=True)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(1,1,out_channels), requires_grad=True)

        self.use_bias = use_bias

    def forward(self, x):        
        x = torch.matmul(x,self.matrix) 
        if self.use_bias:
            x = x+ self.bias 
        return x
        
# Implements RNN Unit


# Fill in the missing lines of code in the RNN and GRU cells functions. The RNN contains some lines which you may find
# helpful (or choose to omit and implement on your own). The gates and update operators should consist of a single FC
# layer (hidden state dim. should be between 64-128 for the lowest test error). The convention of the tensors for these
# recurrent networks is: batch element x “time” x feature vector. So the recurrence (your iteration in the code) should
# apply on the second axis. Once the review is parsed, its hidden-state should pass through an MLP which produces the
# final output sentiment prediction (a 2-class one hot vector).
# Run each of these two recurrent network architectures, describe your experiments with two different hidden-state
# dimensions, and the train/test accuracies obtained (with plots). Explain what could lead to the different results
# you found in the experiment (both RNN vs GRU and hidden state size). Come up with a test review of your own which
# demonstrates the different capabilities of the two recurrent models. Add this review (and maybe variations of it
# that you’ve experimented with), the results obtained and your explanation.

class ExRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExRNN, self).__init__()

        self.hidden_size = hidden_size
        self.sigmoid = torch.sigmoid

        # RNN Cell weights
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden2out = nn.Linear(hidden_size, output_size)

    def name(self):
        return "RNN"

    def forward(self, x, hidden_state):

        # Implementation of RNN cell

        combined = torch.cat((x, hidden_state), 1)
        hidden = self.sigmoid(self.in2hidden(combined))
        output = self.hidden2out(hidden)

        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)

# Implements GRU Unit

class ExGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExGRU, self).__init__()
        self.hidden_size = hidden_size
        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh

        # GRU Cell weights
        # the in2hidden layer is used to generate the update gate
        # the hidden2out layer is used to generate the reset gate
        # the W_z, W_r and W_h layers are used to generate the hidden state
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden2out = nn.Linear(hidden_size, output_size)

        self.W_z = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_r = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_h = MatMul(hidden_size, hidden_size, use_bias=False)


    def name(self):
        return "GRU"

    def forward(self, x, hidden_state):

        # Implementation of GRU cell

        combined = torch.cat((x, hidden_state), 1)
        z = self.sigmoid(self.in2hidden(combined))
        r = self.sigmoid(self.in2hidden(combined))
        h = self.tanh(self.W_h(torch.mul(r, hidden_state)))
        hidden = torch.mul(z, hidden_state) + torch.mul(1-z, h)
        output = self.hidden2out(hidden)

        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)


class ExMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExMLP, self).__init__()

        self.ReLU = torch.nn.ReLU()

        # Token-wise MLP network weights
        self.layer1 = MatMul(input_size,hidden_size)
        # additional layer(s) ...
        self.layer2 = MatMul(hidden_size,hidden_size)
        self.layer3 = MatMul(hidden_size,output_size)



        

    def name(self):
        return "MLP"

    def forward(self, x):

        # Token-wise MLP network implementation
        
        x = self.layer1(x)
        x = self.ReLU(x)
        # rest
        x = self.layer2(x)
        x = self.ReLU(x)
        x = self.layer3(x)

        return x


class ExLRestSelfAtten(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExLRestSelfAtten, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        self.ReLU = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(2)
        
        # Token-wise MLP + Restricted Attention network implementation

        self.layer1 = MatMul(input_size,hidden_size)
        self.W_q = MatMul(hidden_size, hidden_size, use_bias=False)
        # rest ...


    def name(self):
        return "MLP_atten"

    def forward(self, x):

        # Token-wise MLP + Restricted Attention network implementation

        x = self.layer1(x)
        x = self.ReLU(x)

        # generating x in offsets between -atten_size and atten_size 
        # with zero padding at the ends

        padded = pad(x,(0,0,atten_size,atten_size,0,0))

        x_nei = []
        for k in range(-atten_size,atten_size+1):
            x_nei.append(torch.roll(padded, k, 1))

        x_nei = torch.stack(x_nei,2)
        x_nei = x_nei[:,atten_size:-atten_size,:]
        
        # x_nei has an additional axis that corresponds to the offset

        # Applying attention layer

        # query = ...
        # keys = ...
        # vals = ...


        return x, atten_weights


# prints portion of the review (20-30 first words), with the sub-scores each work obtained
# prints also the final scores, the softmaxed prediction values and the true label values

def print_review(rev_text, sbs1, sbs2, lbl1, lbl2):

        print(rev_text)
        print("Sub-scores:")
        print(sbs1)
        print(sbs2)
        print("Final scores:")
        print(sbs1.mean())
        print(sbs2.mean())
        print("Softmaxed predictions:")
        print(torch.nn.Softmax(0)(torch.tensor([sbs1.mean(), sbs2.mean()])))
        print("True labels:")
        print(lbl1)
        print(lbl2)


# select model to use

if run_recurrent:
    if use_RNN:
        model = ExRNN(input_size, output_size, hidden_size)
    else:
        model = ExGRU(input_size, output_size, hidden_size)
else:
    if atten_size > 0:
        model = ExLRestSelfAtten(input_size, output_size, hidden_size)
    else:
        model = ExMLP(input_size, output_size, hidden_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print the device used
print("Using device: " + str(device))
print("Using model: " + model.name())

if reload_model:
    print("Reloading model")
    model = model.to(device)
    # model.load_state_dict(torch.load(model.name() + ".pth", map_location=device))
    model = torch.load(model.name() + ".pth")
    model.eval()

    # print the results of the model on the ld.my_test_texts and compare them to ld.my_test_labels
    if run_recurrent:
        for i in range(len(ld.my_test_texts)):
            review = ld.my_test_texts[i]
            label = ld.my_test_labels[i]
            review = torch.tensor(ld.preprocess_review(review)).to(device)
            hidden_state = model.init_hidden(1)
            for j in range(num_words):
                output, hidden_state = model(review[:,j,:], hidden_state)
            final_output = model.hidden2out(hidden_state)
            print("Review: " + ld.my_test_texts[i])
            print("Prediction: " + str(output))
            print("True label: " + str(label))
            print("")

    else:
        for i in range(len(ld.my_test_texts)):
            review = ld.my_test_texts[i]
            label = ld.my_test_labels[i]
            review = torch.tensor(ld.preprocess_review(review)).to(device)
            output = model(review)
            print("Review: " + ld.my_test_texts[i])
            print("Prediction: " + str(output))
            print("True label: " + str(label))
            print("")



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss = 1.0
test_loss = 1.0
train_losses = []
test_losses = []
epoch_train_losses = []
epoch_test_losses = []
model = model.to(device)

reviews_list = [
    ['ah', 'hitchcock', 'it', 'hard', 'to', 'find', 'bad', 'hitchcock', 'movie', 'until', 'he', 'lost', 'it', 'after', 'the', 'birds', 'and', 'saboteur', 'proves', 'the', 'point', 'having', 'admired', 'most', 'of', 'this', 'director', 'work', 'for', 'many', 'years', 'had', 'managed', 'to', 'skip', 'this', 'one', 'perhaps', 'from', 'lack', 'of', 'interest', 'in', 'priscilla', 'lane', 'and', 'robert', 'cummings', 'as', 'lead', 'actors', 'was', 'of', 'course', 'familiar', 'with', 'the', 'statue', 'of', 'liberty', 'climax', 'from', 'having', 'seen', 'it', 'repeatedly', 'in', 'film', 'retrospectives', 'but', 'wrongly', 'assumed', 'the', 'story', 'leading', 'up', 'to', 'it', 'might', 'not', 'hold', 'my', 'interest', 'was', 'wrong', 'the', 'suspenseful', 'plot', 'gets', 'cooking', 'right', 'off', 'the', 'bat', 'through', 'chance', 'encounter', 'between', 'the', 'bad']
    ,
    ['whoa', 'mean', 'whoa', 'mean', 'whoa', 'whoa', 'br', 'br', 'saw', 'this', 'movie', 'waaay', 'back', 'when', 'was', 'eight', 'in', 'back', 'then', 'cgi', 'films', 'were', 'rarity', 'and', 'good', 'ones', 'even', 'more', 'so', 'also', 'back', 'then', 'we', 'listened', 'to', 'things', 'called', 'cd', 'players', 'but', 'digress', 'used', 'to', 'like', 'this', 'movie', 'lot', 'way', 'back', 'then', 'and', 'up', 'till', 'viewing', 'it', 'again', 've', 'held', 'reaally', 'fond', 'memories', 'of', 'it', 'hey', 'it', 'don', 'bluth', 'anyone', 'who', 'hates', 'all', 'dogs', 'go', 'to', 'heaven', 'is', 'clearly', 'robot', 'but', 'again', 'digress', 'br', 'br', 'then', 'saw', 'it', 'again', 'this', 'really', 'isn', 'one', 'of', 'his', 'best', 'can', 'say', 'now', 'eleven', 'years', 'later']
    ,
    ['i', 'would', 'not', 'compare', 'it', 'to', 'le', 'placard', 'which', 'imho', 'had', 'more', 'comic', 'moments', 'but', 'romuald', 'juliette', 'while', 'being', 'slow', 'starter', 'certainly', 'kept', 'your', 'attention', 'going', 'throughout', 'the', 'film', 'nicely', 'paced', 'and', 'reaching', 'heart', 'warming', 'conclusion', 'there', 'were', 'many', 'marvellous', 'comedic', 'moments', 'some', 'brilliant', 'pathos', 'and', 'realistic', 'situation', 'acting', 'by', 'all', 'actors', 'br', 'br', 'it', 'was', 'typically', 'french', 'film', 'in', 'which', 'while', 'confronting', 'prejudices', 'and', 'phobias', 'which', 'in', 'turn', 'the', 'made', 'the', 'viewer', 'confront', 'his', 'own', 'shortcomings', 'am', 'certainly', 'pleased', 'to', 'have', 'this', 'in', 'my', 'library', 'and', 'will', 'no', 'doubt', 'watch', 'it', 'time', 'and', 'time', 'again', 'which', 'to', 'me', 'is']
    ,
    ['maybe', 'this', 'was', 'an', 'important', 'movie', 'and', 'that', 'why', 'people', 'rank', 'it', 'so', 'highly', 'but', 'honestly', 'it', 'isn', 'very', 'good', 'in', 'hindsight', 'it', 'easy', 'to', 'see', 'that', 'chaplin', 'probably', 'all', 'of', 'hollywood', 'was', 'incredibly', 'naive', 'about', 'the', 'magnitude', 'of', 'what', 'was', 'really', 'going', 'on', 'in', 'the', 'ghettos', 'so', 'you', 'can', 'fault', 'him', 'too', 'much', 'for', 'the', 'disconnect', 'that', 'affects', 'modern', 'viewer', 'but', 'the', 'disconnect', 'remains', 'br', 'br', 'more', 'disappointingly', 'the', 'movie', 'is', 'just', 'clunky', 'it', 'as', 'if', 'chaplin', 'had', 'no', 'idea', 'that', 'movies', 'had', 'progressed', 'in', 'sophistication', 'since', 'the', 'silent', 'era', 'the', 'set', 'pieces', 'those', 'involving', 'both', 'the', 'jewish', 'barber']
    ]
def check_and_print_reviews(dataset, reviews_list):
    for labels, reviews, reviews_text in dataset:
        for review_text in reviews_text:
            review_words = review_text[:10]  # review_text is already a list of words
            for target_list in reviews_list:
                if review_words == target_list:
                    print("Full review:", ' '.join(review_text))  # Join the list of words into a single string
                    break

# Check train dataset
print("Checking train dataset:")
check_and_print_reviews(train_dataset, reviews_list)

# Check test dataset
print("Checking test dataset:")
check_and_print_reviews(test_dataset, reviews_list)

#training steps in which a test step is executed every test_interval


for epoch in range(num_epochs):

    itr = 0 # iteration counter within each epoch
    temp_train_losses = []
    temp_test_losses = []


    for labels, reviews, reviews_text in train_dataset:   # getting training batches
        labels = labels.to(device)
        reviews = reviews.to(device)


        itr = itr + 1

        if (itr + 1) % test_interval == 0:
            test_iter = True
            labels, reviews, reviews_text = next(iter(test_dataset)) # get a test batch
        else:
            test_iter = False

        # Recurrent nets (RNN/GRU)

        if run_recurrent:
            hidden_state = model.init_hidden(int(labels.shape[0]))

            for i in range(num_words):
                output, hidden_state = model(reviews[:,i,:], hidden_state)  # HIDE

            final_output = model.hidden2out(hidden_state)
        else:

        # Token-wise networks (MLP / MLP + Atten.)

            sub_score = []
            if atten_size > 0:
                # MLP + atten
                sub_score, atten_weights = model(reviews)
            else:
                # MLP
                sub_score = model(reviews)

            final_output = torch.mean(sub_score, 1)

        # cross-entropy loss

        loss = criterion(final_output, labels)

        # optimize in training iterations

        if not test_iter:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # averaged losses
        if test_iter:
            test_loss = 0.8 * float(loss.detach()) + 0.2 * test_loss
            temp_test_losses.append(test_loss)
        else:
            train_loss = 0.9 * float(loss.detach()) + 0.1 * train_loss
            temp_train_losses.append(train_loss)

        if test_iter:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Step [{itr + 1}/{len(train_dataset)}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Test Loss: {test_loss:.4f}"
            )
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            if not run_recurrent:
                nump_subs = sub_score.detach().numpy()
                labels = labels.detach().numpy()
                print_review(reviews_text[0], nump_subs[0,:,0], nump_subs[0,:,1], labels[0,0], labels[0,1])

            # saving the model
            torch.save(model, model.name() + ".pth")

    epoch_train_loss = sum(temp_train_losses) / len(temp_train_losses) if len(temp_train_losses)!=0 else 0
    epoch_test_loss = sum(temp_test_losses) / len(temp_test_losses) if len(temp_test_losses)!=0 else 0
    epoch_train_losses.append(epoch_train_loss)
    epoch_test_losses.append(epoch_test_loss)




# Plotting the losses with the current model name for each epoch

import matplotlib.pyplot as plt
# plot the losses for each epoch
plt.plot(epoch_train_losses, label='train')
plt.plot(epoch_test_losses, label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title(model.name())
plt.show()
