import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import smooth

np.set_printoptions(precision=4)

cuda = torch.cuda.is_available()
if cuda:
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    
    
class Adapter(nn.Module):
    """
    A network to perform domain adaptation, converting the handwritten digit space to the computer-generated digit space.
    """
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(28*28, 128),
                                 nn.LeakyReLU(),
                                 nn.Linear(128, 28*28))
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        cache = x
        x = self.seq(x) + cache
        return x
    

class Discriminator(nn.Module):
    """
    A network to discriminate bewteen handwritten and computer-generated digits.
    Outputs the probability that a digit is computer-generated.
    """
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(28*28, 128),
                                 nn.LeakyReLU(),
                                 nn.Linear(128, 1),
                                 nn.Sigmoid())
    
    def forward(self, x):
        return self.seq(x)

class Classifier(nn.Module):
    """
    An output to classify a digit passed into it. Trained on the adapter output of the computer-generated examples.
    
    After the adapter is trained, it should give a good evaluation on the handwritten digits.
    """
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(28*28, 128),
                                 nn.LeakyReLU(),
                                 nn.Linear(128, 10))
        
    def forward(self, x):
        return self.seq(x)
    
class Net():
    """
    A net to classify digits, both handwritten and computer-generated.
    
    Suppose the only labeled data you have is for the computer-generated digits. This net consists of three neural nets:
    - an adapter to convert both the computer-generated and handwritten digits into the same space
    - a discriminator to adversarially discriminate between the adapter's output of handwritten and computer-generated digits
    - a classifier to classify the digit represented by the adapter's output
    Without the labels for the handwritten digits, you train the classifier only on the labeled computer-generated digits.
    
    The Net constructor takes as arguments learning rates for the adapter, discriminator, and classifier.
    """
    def __init__(self, adapter_lr=3e-6, discriminator_lr=3e-6, classifier_lr=3e-6):
        # loss functions for discriminator and classifier output
        self.loss_fn_D = nn.BCELoss()
        self.loss_fn_C = nn.CrossEntropyLoss()
        
        # store the losses for graphing later
        self.a_computer_losses = []
        self.a_handwritten_losses = []
        self.d_computer_losses = []
        self.d_handwritten_losses = []
        self.c_losses = []

        # initialize the three nets
        self.adapter = Adapter()
        self.discriminator = Discriminator()
        self.classifier = Classifier()
        if cuda:
            self.adapter.cuda()
            self.discriminator.cuda()
            self.classifier.cuda()

        # initialize the optimizers
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=discriminator_lr, betas=(0.5, 0.999))
        self.optimizer_A = torch.optim.Adam(self.adapter.parameters(), lr=adapter_lr, betas=(0.5, 0.999))
        self.optimizer_C = torch.optim.Adam(self.classifier.parameters(), lr=classifier_lr, betas=(0.5, 0.999))

    def fit(self, num_iterations, images, labels, batch_size=1024, print_every=100):
        """
        This class fits the model and displays progress.
        num_iterations: number of iterations to train for
        images: a dict of dicts:
            images['computer'] = {'train': x, 'test': y}
                where x and y are the train and test computer-generated images
            images['handwritten'] = {'train': x, 'test': y}
                where x and y are the train and test handwritten images
        labels: a dict of dicts:
            labels['computer'] = {'train': x, 'test': y}
                where x and y are the train and test computer-generated labels
            labels['handwritten'] = {'train': x, 'test': y}
                where x and y are the train and test handwritten labels
        batch_size: training batch size
        print_every: how often to output progress info, in number of iterations
        
        Note that the handwritten labels are not used during training, as this net is an exercise in domain adaptation.
        
        The progress output prints:
        - the loss for the adapter, discriminator, and classifier
        - the classifier score on each of the two test datasets
        - the discriminator score on each digit for both the train and test sets
        - graphs for the loss values for the adapter, discriminator, and classifier
        - graphs representing the adapter outputs of twenty digits (0 through 9, randomly sampled from both test sets),
            the handwritten and computer-generated outputs superimposed on one another
        """
        for i in range(num_iterations):
            self.adapter.train()
            self.classifier.train()
            self.discriminator.train()

            # batch images
            computer_batch_indices = np.random.randint(images['computer']['train'].shape[0], size=batch_size)
            handwritten_batch_indices = np.random.randint(images['handwritten']['train'].shape[0], size=batch_size)
            computer_images = FloatTensor(images['computer']['train'][computer_batch_indices]).view(-1, 1, 28, 28)
            computer_labels = LongTensor(labels['computer']['train'][computer_batch_indices])
            handwritten_images = FloatTensor(images['handwritten']['train'][handwritten_batch_indices]).view(-1, 1, 28, 28)

            # adversarial ground truths
            computer_truth_val = FloatTensor(batch_size, 1).fill_(1.0)
            computer_truth_val.requires_grad = False
            handwritten_truth_val = FloatTensor(batch_size, 1).fill_(0.0)
            handwritten_truth_val.requires_grad = False

            # forward pass for adapter parameters
            self.optimizer_A.zero_grad()
            adapter_out_computer = self.adapter(computer_images)
            classifier_out = self.classifier(adapter_out_computer)
            discriminator_out_computer = self.discriminator(adapter_out_computer)
            loss_C_A = self.loss_fn_C(classifier_out, computer_labels)
            loss_A_computer = self.loss_fn_D(discriminator_out_computer, computer_truth_val)
            self.a_computer_losses.append(loss_A_computer.item())
            
            adapter_out_handwritten = self.adapter(handwritten_images)
            discriminator_out_handwritten = self.discriminator(adapter_out_handwritten)
            loss_A_handwritten = self.loss_fn_D(discriminator_out_handwritten, computer_truth_val) # want the discriminator to think the adapted handwritten image is computer-generated
            self.a_handwritten_losses.append(loss_A_handwritten.item())

            # backprop for adapter
            loss_A = (loss_A_computer + loss_A_handwritten)/2
            loss_A.backward(retain_graph=True)
            self.optimizer_A.step()

            # forward pass for discriminator parameters
            self.optimizer_D.zero_grad()
            adapter_out_computer = self.adapter(computer_images).detach()
            classifier_out = self.classifier(adapter_out_computer)
            discriminator_out_computer = self.discriminator(adapter_out_computer)
            loss_C_D = self.loss_fn_C(classifier_out, computer_labels)
            loss_D_computer = self.loss_fn_D(discriminator_out_computer, computer_truth_val)
            self.d_computer_losses.append(loss_D_computer.item())

            adapter_out_handwritten = self.adapter(handwritten_images).detach()
            discriminator_out_handwritten = self.discriminator(adapter_out_handwritten)
            loss_D_handwritten = self.loss_fn_D(discriminator_out_handwritten, handwritten_truth_val)
            self.d_handwritten_losses.append(loss_D_handwritten.item())

            # backprop for discriminator
            loss_D = (loss_D_computer + loss_D_handwritten)/2
            loss_D.backward(retain_graph=True)
            self.optimizer_D.step()

            # backprop for classifier
            self.optimizer_C.zero_grad()
            loss_C = (loss_C_A + loss_C_D)/2
            self.c_losses.append(loss_C.item())
            loss_C.backward()
            self.optimizer_C.step()

            # print progress
            if i % print_every == 0:
                
                # print losses and accuracies
                X_computer = FloatTensor(images['computer']['test']).reshape(-1, 1, 28, 28)
                y_computer = LongTensor(labels['computer']['test'])
                X_handwritten = FloatTensor(images['handwritten']['test']).reshape(-1, 1, 28, 28)
                y_handwritten = LongTensor(labels['handwritten']['test'])
                computer_score = self.score(X_computer, y_computer)
                handwritten_score = self.score(X_handwritten, y_handwritten)
                lc = round(loss_C.item(), 4)
                la = round(loss_A.item(), 4)
                ld = round(loss_D.item(), 4)
                cs = round(computer_score, 4)
                hs = round(handwritten_score, 4)
                print(f'i: {i}, loss_C = {lc}, loss_A = {la}, loss_D = {ld}, comp_score = {cs}, mnist_score = {hs}')
                
                # print loss graphs
                if i != 0:
                    colors = [('#004f11', '#73ff91'), ('#386ed9', '#95b5f5'), ('#d10000')]
                    arrs = [(self.a_computer_losses, self.a_handwritten_losses),
                            (self.d_computer_losses, self.d_handwritten_losses),
                            (self.c_losses)]
                    names = [('Adapter Loss (Computer)', 'Adapter Loss (Handwritten)'),
                             ('Discriminator Loss (Computer)','Discriminator Loss (Handwritten)'),
                             ('Classifier Loss',)]
                    _, axs = plt.subplots(1, 3, figsize=(16, 4))
                    for j in range(3):
                        for arr, color, name in zip(arrs[j], colors[j], names[j]):
                            axs[j].plot(arr, alpha=0.3, color=color)
                            axs[j].plot(smooth(arr, print_every), color=color, label=name)
                        axs[j].legend()
                    axs[0].set_ylim(0, 2)
                    axs[1].set_ylim(0, 2)
                    axs[2].set_ylim(0, 3)
                    plt.show()

                    # print outputs of adapter
                    _, axs2 = plt.subplots(2, 5, figsize=(16, 4))
                    for j in range(10):
                        computer_subset = images['computer']['test'][labels['computer']['test'] == j]
                        handwritten_subset = images['handwritten']['test'][labels['handwritten']['test'] == j]
                        n = np.random.randint(computer_subset.shape[0])
                        m = np.random.randint(handwritten_subset.shape[0])
                        computer_example = FloatTensor(computer_subset[n].reshape(1, 1, 28, 28))
                        handwritten_example = FloatTensor(handwritten_subset[m].reshape(1, 1, 28, 28))
                        with torch.no_grad():
                            self.adapter.eval()
                            computer_adapter_out = self.adapter(computer_example).cpu().numpy().reshape(-1)
                            handwritten_adapter_out = self.adapter(handwritten_example).cpu().numpy().reshape(-1)
                            axs2[j // 5, j % 5].plot(computer_adapter_out, alpha=0.3, color='b')
                            axs2[j // 5, j % 5].plot(handwritten_adapter_out, alpha=0.3, color='r')
                            self.adapter.train()
                    plt.show()

                    # print discriminator scores
                    train_discriminator_scores, test_discriminator_scores = self.get_discriminator_scores(images, labels)

                    print('\nTraining Set Discriminator Scores:')
                    for j in range(10):
                        computer_score = round(train_discriminator_scores[j]['computer'], 4)
                        handwritten_score = round(train_discriminator_scores[j]['handwritten'], 4)
                        print(f'{j}   comp score: {computer_score}, hand score: {handwritten_score}')

                    print('\nTest Set Discriminator Scores:')
                    for j in range(10):
                        computer_score = round(test_discriminator_scores[j]['computer'], 4)
                        handwritten_score = round(test_discriminator_scores[j]['handwritten'], 4)
                        print(f'{j}   comp score: {computer_score}, hand score: {handwritten_score}')

                    print('\n')



    def predict(self, X):
        """
        Predict which digit the images in X represent.
        Here X is a batch of grayscale 28x28 images, represented in a torch.FloatTensor of shape (N, 1, 28, 28).
        """
        self.adapter.eval()
        self.classifier.eval()
        with torch.no_grad():
            preds = self.classifier(self.adapter(X)).argmax(axis=1)
        self.adapter.train()
        self.classifier.train()
        return preds.cpu().numpy()

    def score(self, X, y):
        """
        Returns the accuracies of the predictions of the images of X, where y is the true labels.
        Here X is a batch of grayscale 28x28 images, represented in a torch.FloatTensor of shape (N, 1, 28, 28)
        and y is a torch.LongTensor of shape (N,)
        """
        preds = self.predict(X)
        with torch.no_grad():
            y = y.cpu().numpy()
            score = (preds == y).mean()
        return score

    def get_discriminator_scores(self, images, labels):
        """
        Utility function used in self.fit.
        
        Outputs the accuracy of the discriminator on test and train sets and also on each of the ten kinds of digits.
        """
        self.adapter.eval()
        self.discriminator.eval()
        scores = {'train': [], 'test': []}
        for phase in ('train', 'test'):
            for i in range(10):
                computer_subset = FloatTensor(images['computer'][phase][labels['computer'][phase] == i].reshape(-1, 1, 28, 28))
                handwritten_subset = FloatTensor(images['handwritten'][phase][labels['handwritten'][phase] == i].reshape(-1, 1, 28, 28))
                with torch.no_grad():
                    computer_out = self.discriminator(self.adapter(computer_subset)).cpu().numpy()
                    computer_preds = (computer_out > 0.5).astype(int) # predict which ones are computer-generated
                    computer_acc = (computer_preds == 1).mean() # all should be 1 (computer)
                    handwritten_out = self.discriminator(self.adapter(handwritten_subset)).cpu().numpy()
                    handwritten_preds = (handwritten_out > 0.5).astype(int) # predict which ones are computer-generated
                    handwritten_acc = (handwritten_preds == 0).mean() # all should be 0 (handwritten)
                    scores[phase].append({'computer': computer_acc, 'handwritten': handwritten_acc})
        self.adapter.train()
        self.discriminator.train()
        return scores['train'], scores['test']