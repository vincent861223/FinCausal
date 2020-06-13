from data_loader.data_loaders import FinCausalDataset, FinCausalDataloader
if __name__ == "__main__":
    # dataset = FinCausalDataset("data/train.csv", train=True)
    # print(dataset[0])
    # dataset = FinCausalDataset("data/test.csv", test=True)

    dataloader = FinCausalDataloader("data/train.csv", batch_size=16)
    for i, batch in enumerate(dataloader):
        if i == 0:
            print(batch)
            break


