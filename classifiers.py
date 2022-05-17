import torch
from clip import clip


def zeroshot_classifier(classnames:list, templates:list, model:clip.model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            text = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = model.encode_texts(texts) # embed
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings /= class_embeddings.norm()
            zeroshot_weights.append(class_embeddings)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights
