import numpy as np
def shows_attention1(att_maps, pos = [28, 122]):
    att_map1 = att_maps[0]
    _, _, h, w = att_map1.shape
    vis_map = np.zeros((h, w), dtype=np.float32)
    att_vector = att_map1[0,:,pos[0], pos[1]]
    for i in range(w):
        vis_map[pos[0], i] = att_vector[i]
    for i in range(w,h+w-1):
        new_i = i - w
        if i >= pos[0]:
            new_i = new_i + 1
        vis_map[new_i, pos[1]] = att_vector[i]
    return vis_map

def shows_attention2(att_maps, pos = [28, 122]):
    att_map1 = att_maps[0]
    att_map2 = att_maps[1]
    _, _, h, w = att_map1.shape
    vis_map = np.zeros((h, w), dtype=np.float32)

    att_vector = att_map2[0,:,pos[0], pos[1]]
    for i in range(w):
        map_step1 = shows_attention1(att_maps, pos=[pos[0], i])
        vis_map += att_vector[i] * map_step1
    for i in range(w,h+w-1):
        new_i = i - w
        if i >= pos[0]:
            new_i = new_i + 1
        map_step1 = shows_attention1(att_maps, pos=[new_i, pos[1]])
        vis_map += att_vector[i] * map_step1
    return vis_map

def make_image(vis_map, outputname):
    import matplotlib.pyplot as plt
    fig = plt.imshow(vis_map, cmap='hot', interpolation='bilinear')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.margins(0,0)
    plt.savefig(outputname)

# att_maps is the attention map of RCCA
att_maps = [att.data.cpu().numpy() for att in att_maps]
vis_map = shows_attention2(att_maps, [19, 70])
make_image(vis_map, 'attention_vis.png')