# Packet Routing Using Multi Agent DQN and Single Agent GCN

Sai Shreyas Bhavanasi, Lorenzo Pappone, Dr. Flavio Esposito

This repo contains the code for the paper 'Dealing with Changes: Resilient Routing via Graph Neural Networks and Multi-Agent Deep Reinforcement Learning' submitted to the IEEE TNSM (Special issue on Reliable Networks)

To run the models, simply run the command `python train.py`

This will run all the models: MA-DQN, SA-GCN, SPF, and ECMP on a 50 Node Barabasi network. The networks are genreated via BRITE topology generator.

To install the required dependencies, the following command can be run:

`pip install -r requirements.txt`

### Citing this repo

If you use this repo in your research, please cite using the following BibTeX entry:


```BibTeX
@misc{bhavanasi2023-routing-drl,
  author =       {Sai Shreyas Bhavanasi and Lorenzo Pappone and Flavio Esposito},
  title =        {Dealing with Changes: Resilient Routing via Graph Neural Networks and Multi-Agent Deep Reinforcement Learning},
  howpublished = {\url{https://github.com/routing-drl/main}},
  year =         {2023}
}
```