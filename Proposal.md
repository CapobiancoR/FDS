# A few ideas

the dataset was already preprocessed, anyways i checked for img sizes and then i normalized and augmented it again, i also shown the basic functioning of the class used, anyway you can find documentation online, is a tensorflow framework.

would be cool to train on both datasets, the "standard" one and the one augmented a second time to analyze differences in performance,
also I think that if we can incorporate more dataset togheter (leading to more preprocessing work) we can achieve better results, as shown in several studies.

Furthermore testing the model on a different dataset then the one used in the paper, maybe with a fine tuning of few layers on the new dataset, would be a cool comparison that would let us analyze deeper the generalization of the model.

each trainig should be done with the base model of the paaper and with the one we're proposing.

## So what i mean

base model:
    -train on base dataset (just replicating the paper result)
    -train on augmented dataset
    -train on join dataset (using more dataset togheter)

our model:
    -train on base dataset (just replicating the paper result)
    -train on augmented dataset
    -train on join dataset (using more dataset togheter)

## Analysis

would be cool to test (not train), all 6 previous checkpoints, on a different dataset then the one used in training, we could create (or search if already exist) a generalization score based on how well it performs on new photos that do not have the biases of the training dataset.

For the model in which we use all dataset we caan actually just take a pic of a plant (a type of classified plant) aand sumbit it to the model


### Note
It's all just 1 implementation, starts replicating the model, train with the same function 3 times, modify a bit the model to improve performances and train again 3 times