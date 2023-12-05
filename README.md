# Deep-Chess
Modified version of [Deep-Chess](https://arxiv.org/pdf/1711.09667.pdf) paper. 


## Dataset
The `scrape_games.py` script was employed to fetch and filter chess games featuring players with an Elo rating surpassing 2000. The source for these games is [https://computerchess.org.uk/ccrl/4040/](https://computerchess.org.uk/ccrl/4040/). Games that resulted in a draw were excluded, resulting in a dataset of around 550,000 games. For each game, 15 positions were extracted, with the condition that they couldn't originate from the first 5 moves. These positions were then transformed into bitboards, each with a size of 773 bits.


## Models
Instead of using a Deep Belief Network (DBN), I employed a regular autoencoder (AE) with a Leaky ReLU activation function. The AE was initially trained to extract essential features from bitboards, employing an encoder architecture of 773-600-400-200-100. Subsequently, the extracted features were concatenated and fed into the Deep-Chess (DC) model of architecture 200-400-200-100-2. This model compares positions and outputs the superior one from white's perspective. Deep-Chess uses modified version of minimax algorithm with alpha-beta prunnig to determine best move. Instead of asigning to each position numerical value, model directly compares positions and outputs the best one.


## Training
For training the autoencoder (AE), one million black and one million white positions were set aside. An additional 100 thousand positions for both white and black were reserved for validation purposes. The AE underwent training for 200 epochs with a learning rate of 0.005, decaying by a factor of 0.98 after each epoch, and a batch size of 512. After training, the AE exhibited an accuracy of 93% on the training set and 92% on the validation set.

During DC training, one position where white won (W) and one where black won (L) were sampled and placed either as (W, L) or (L, W). The model's task was to output (1, 0) or (0, 1). DC model was trained on the entire dataset for 140 epochs with a learning rate of 0.01, decaying by a factor of 0.99 after each epoch. The validation loss plateaued around the 100th epoch, resulting in final accuracies of 93% on the training set and 90% on the validation set.


## Results
To be honest, the obtained results fell short of expectations. Despite experimenting with various hyperparameters, I was unable to surpass a 90% accuracy on the validation set. Interestingly, the authors of the paper claimed to achieve training and validation accuracies of 98.2% and 98.0%, respectively. However, after thorough research, no concrete evidence or results were found to corroborate their claims. It's worth noting that this disparity could potentially be attributed to the use of a regular autoencoder instead of a DBN as suggested in the paper. Optimal value for depth in minimax is 3, since time required for search rises exponentialy.


## Run Locally
Ensure you have Python 3.10.8 installed and install dependencies from `requirements.txt`.
