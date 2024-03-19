from guidance.if_utils import IF
import argparse
import time
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts_set', type=str, default='vehicle', choices=['vehicle', 'daily_life', 'animal'],
                        help="optimizer")
    opt = parser.parse_args()
    return opt




if __name__ == '__main__':
    opt = parse_args()
    device = 'cuda'
    guidance = IF(device, False, [0.02, 0.98])
    
    
    if opt.prompts_set == 'vehicle':
        texts_ = []
        texts = []
        with open('./vehicle.txt', 'r') as f:
            texts_ = f.readlines()
        for text in texts_:
            if len(text.strip().split(' ')) < 75 and len(text.strip().split(' ')) > 1:
                texts.append(text.strip().strip('.'))
        print(len(texts))
        texts_embeddings = []
        i = 0
        t = time.time()
        for text in texts:
            embeddings = {}
            embeddings['default'] = guidance.get_text_embeds([text]).cpu()
            for d in ['front', 'side', 'back']:
                embeddings[d] = guidance.get_text_embeds([f"{text}, {d} view"]).cpu()
            texts_embeddings.append(embeddings)
            i = i+1
            if i % 100 == 0:
                print(f'finish {i} text embeddings, time: {time.time() - t}')
                t = time.time()
        torch.save(texts_embeddings, './vehicle_if.pkl')
        quit()
    
    elif opt.prompts_set == 'daily_life':
        texts_ = []
        texts = []
        with open('./daily_life.txt', 'r') as f:
            texts_ = f.readlines()
        for text in texts_:
            if len(text.strip().split(' ')) < 75:
                texts.append(text.strip())

        print(len(texts))
        texts_embeddings = []
        i = 0
        t = time.time()
        for text in texts:
            embeddings = {}
            embeddings['default'] = guidance.get_text_embeds([text]).cpu()
            for d in ['front', 'side', 'back']:
                embeddings[d] = guidance.get_text_embeds([f"{text}, {d} view"]).cpu()
            texts_embeddings.append(embeddings)
            i = i+1
            if i % 100 == 0:
                print(f'finish {i} text embeddings, time: {time.time() - t}')
                t = time.time()
        torch.save(texts_embeddings, './daily_life_if.pkl')
        
    elif opt.prompts_set == 'animal':
        texts = []
        species = ['wolf', 'dog', 'panda', 'fox', 'civet', 'cat', 'red panda', 'teddy bear', 'rabbit', 'koala']
        item = ['in a bathtub', 'on a stone', 'on books', 'on a table', 'on the lawn', 'in a basket', 'null']
        gadget = ['a tie', 'a cape', 'sunglasses', 'a scarf', 'null']
        hat = ['beret', 'beanie', 'cowboy hat', 'straw hat', 'baseball cap', 'tophat', 'party hat', 'sombrero', 'null']
        for s in species:
            for i in item:
                for g in gadget:
                    for h in hat:
                        if i == 'null':
                            texts.append(f'a {s} wearing {g} and wearing a {h}')
                        elif g == 'null':
                            texts.append(f'a {s} sitting {i} and wearing a {h}')
                        elif h == 'null':
                            texts.append(f'a {s} sitting {i} and wearing {g}')
                        else:
                            texts.append(f'a {s} sitting {i} and wearing {g} and wearing a {h}')
        texts_embeddings = []
        i = 0
        t = time.time()
        for text in texts:
            embeddings = {}
            embeddings['default'] = guidance.get_text_embeds([text]).cpu()
            for d in ['front', 'side', 'back']:
                embeddings[d] = guidance.get_text_embeds([f"{text}, {d} view"]).cpu()
            texts_embeddings.append(embeddings)
            i = i+1
            if i % 100 == 0:
                print(f'finish {i} text embeddings, time: {time.time() - t}')
                t = time.time()
        torch.save(texts_embeddings, './animal_if.pkl')