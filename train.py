from helper.graph import compute_flow_value
import networkx as nx
from environment.util import create_graph, get_flows, adjust_lat_band
import os
from environment.env import Env as link_hop_env
from models.DQN import *
import time
from models.GCN import GCN_Agent, GCN
from environment.GCN_env import Env
directory = "training_data"


if not os.path.exists(directory):
    os.makedirs(directory)

# uncomment to use seeds
# torch.manual_seed(1)
# np.random.seed(1)
# random.seed(1)
# torch.cpu.manual_seed(1)

lwr = -1.51  # used for evaluating models based on reward function


def ma():
    t0 = time.time()
    contn = "yes"
    env = link_hop_env(directory +
                       "/" + "50Nodes_wax" + ".csv",  G)

    env.graph = adjust_lat_band(env.graph, flows)
    model = MultiAgent(env)
    model.run(10000)
    eps_run = 1000
    while contn == "yes":
        model.run(eps_run)
        inp = input("continue training?")
        sp = inp.split(" ")
        contn = sp[0]
        if len(sp) > 1:
            eps_run = int(sp[1])

    evlt = ""
    evlt = input("evaluate model?")
    if evlt == "yes":
        model.test(2000)
    model.env.graph.remove_nodes_from(nodes_to_remove)
    retrn = input("retrain?")
    while retrn == "yes":
        model.run(2000)
        retrn = input("retrain?")

    t1 = time.time()
    total = t1-t0
    print(f"Multi Agent {round(total/60, 2)} mins")
    
def gcn():
    num_nodes = 50
    max_neighbors = 50

    environment = Env("training_data/save_file",
                      num_nodes_in_graph=num_nodes, 
                      max_neighbors=max_neighbors,
                      graph = G )

    policy_net = GCN(num_nodes, max_neighbors).to(device)
    target_net = GCN(num_nodes, max_neighbors).to(device)

    gcn = GCN_Agent(
        outputs=num_nodes,
        policy_net=policy_net,
        target_net=target_net,
        num_nodes=num_nodes,
        env=environment
    )
    start_time = time()
    gcn.run(10_000)
    print(f'took {time()-start_time} (s)')



def spf():
    t0 = time.time()
    env = link_hop_env(directory +
                       "/" + "spf_150" + ".csv",  G)
    env.graph = adjust_lat_band(env.graph, flows)
    env.graph.remove_nodes_from(nodes_to_remove)

    good = 0
    bad = 0
    reward = 0
    for _ in range(10000):

        obs, done = env.reset(), False

        path = nx.shortest_path(env.graph, obs[0], obs[1])

        for i in range(1, len(path)):
            action = env.neighbors.index(path[i])
            obs, reward, done, infos = env.step(action)

        if reward == 1.01 or reward == lwr:
            good += 1
        else:
            bad += 1
    t1 = time.time()
    total = t1-t0
    print(f"spf {round(total/60, 2)} mins")
    print(f"spf % Routed: {good / float(good + bad)}")



def ecmp():
    t0 = time.time()
    env = link_hop_env(directory +
                       "/" + "ecmp_150" + ".csv",    G)
    env.graph.remove_nodes_from(nodes_to_remove)

    good = 0
    bad = 0
    reward = 0
    for _ in range(10_000):

        obs, done = env.reset(), False
        # print(obs)

        paths = nx.all_shortest_paths(env.graph, obs[0], obs[1])
        path = []
        b = -1

        for p in paths:
            if compute_flow_value(env.graph, tuple(p)) > b:
                b = compute_flow_value(env.graph, tuple(p))
                path = p

        for i in range(1, len(path)):
            action = env.neighbors.index(path[i])
            obs, reward, done, infos = env.step(action)

        if reward == 1.01 or reward == lwr:
            good += 1
        else:
            bad += 1

    t1 = time.time()
    total = t1-t0
    print(f"ecmp {round(total/60, 2)} mins")
    print(f"ecmp % Routed: {good / float(good + bad)}")



def genrt_flows(num_flows):
    env = link_hop_env(directory +
                       "/" + "flow_genrtr" + ".csv", G)
    flows = get_flows(env.graph, num_flows)
    return flows


G = create_graph(50,   100,   "50nodes.brite")
flows = genrt_flows(70)
nodes_to_remove = []
ma()
gcn()
ecmp()
spf()
