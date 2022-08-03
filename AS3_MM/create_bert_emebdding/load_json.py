import json
# from metadata.dataset_spec import load_dataset_spec
#
# datasec = load_dataset_spec('./splits/dtd_splits.json')
# print(datasec.file_pattern)
# datasec = [datasec]
# total_n_classes = 0
# for specs in datasec:
#     total_n_classes += len(specs.get_classes('train'))
# total_n_classes = 0  # 获取所有的类别数量
# specs_dict = {}


# total_n_classes += len(datasec.get_classes(0))


def load_cu_birds_json(dataspec_json_pth):

    with open(dataspec_json_pth,'r') as load_f:
        load_dict = json.load(load_f)

    train_class_number = load_dict['classes_per_split']['TRAIN']
    valid_class_number = load_dict['classes_per_split']['VALID']
    test_class_number  = load_dict['classes_per_split']['TEST']
    class_names        = list(load_dict['class_names'].values())
    total_class        = len(class_names)
    train_class_names    = []
    val_class_names      = []
    test_class_names     = []
    all_class_names_ori  = {}
    for i in range(0,train_class_number):
        all_class_names_ori['cu_birds-train-'+str(i)] = class_names[i]
        train_class_names.append(class_names[i])
    for i in range(train_class_number,train_class_number+valid_class_number):
        all_class_names_ori['cu_birds-val-'+str(i)] = class_names[i]
        val_class_names.append(class_names[i])
    for i in range(train_class_number+valid_class_number,total_class):
        all_class_names_ori['cu_birds-test-'+str(i)] = class_names[i]
        test_class_names.append(class_names[i])

    new_train_class_name = []
    new_val_class_name   = []
    new_test_class_name  = []

    for name in train_class_names:
        new_train_class_name.append(name.split('.')[-1])
    train_class_names = new_train_class_name

    for name in val_class_names:
        new_val_class_name.append(name.split('.')[-1])
    val_class_names = new_val_class_name

    for name in test_class_names:
        new_test_class_name.append(name.split('.')[-1])
    test_class_names = new_test_class_name

    all_class_names = {}

    for i, name in enumerate(train_class_names):
        name_split = name.split('_')
        new_name = ''
        for j in range(len(name_split)):
            new_name = new_name+name_split[j]
            new_name = new_name +' '
        all_class_names['cu_birds-train-'+str(i)] = new_name

    for i, name in enumerate(val_class_names):
        name_split = name.split('_')
        new_name = ''
        for j in range(len(name_split)):
            new_name = new_name + name_split[j]
            new_name = new_name + ' '

        all_class_names['cu_birds-val-'+str(i+train_class_number)] = new_name

    for i,name in enumerate(test_class_names):
        name_split = name.split('_')
        new_name = ''
        for j in range(len(name_split)):
            new_name = new_name + name_split[j]
            if j == (len(name_split) - 1):
                break
            new_name = new_name + ' '
        all_class_names['cu_birds-test-' + str(i+train_class_number+valid_class_number)] = new_name
    return all_class_names_ori, all_class_names


def get_leaves(nodes):
  """Return a list containing the leaves of the graph defined by nodes."""
  leaves = []
  words  = []
  for n in nodes:
    if not n['children_ids']:
      leaves.append(n['wn_id'])
      words.append(n['words'])

  return leaves,words

def load_ilsvrc_json(dataspec_json_pth):
    with open(dataspec_json_pth, 'r') as load_f:
        load_dict = json.load(load_f)
    train_class_graph = load_dict['split_subgraphs']['TRAIN']
    val_class_graph   = load_dict['split_subgraphs']['VALID']
    test_class_graph  = load_dict['split_subgraphs']['TEST']

    train_class_leaves,train_words = get_leaves(train_class_graph)
    val_class_leaves ,val_words  = get_leaves(val_class_graph)
    test_class_leaves ,test_words = get_leaves(test_class_graph)

    train_class_number = len(train_class_leaves)
    val_class_number   = len(val_class_leaves)
    test_class_number  = len(test_class_leaves)
    total_class        = train_class_number+val_class_number+test_class_number
    # print(train_class_number)
    # print(val_class_number)
    # print(test_class_number)


    all_class_names_ori    = {}
    all_class_names        = {}
    for i in range(train_class_number):

        all_class_names_ori['ilsvrc_2012-train-'+str(i)] = train_class_leaves[i]
        all_class_names['ilsvrc_2012-train-'+str(i)]     = train_words[i]

    for i in range(val_class_number):

        all_class_names_ori['ilsvrc_2012-val-' + str(i+train_class_number)] = val_class_leaves[i]
        all_class_names['ilsvrc_2012-val-' + str(i+train_class_number)]     = val_words[i]

    for i in range(test_class_number):
        all_class_names_ori['ilsvrc_2012-test-' + str(i+train_class_number+val_class_number)] = test_class_leaves[i]
        all_class_names['ilsvrc_2012-test-' + str(i+train_class_number+val_class_number)]     = test_words[i]

    return all_class_names_ori,all_class_names

def load_mscoco_json(dataspec_json_pth):
    with open(dataspec_json_pth, 'r') as load_f:
        load_dict = json.load(load_f)

    train_class_number = load_dict['classes_per_split']['TRAIN']
    valid_class_number = load_dict['classes_per_split']['VALID']
    test_class_number = load_dict['classes_per_split']['TEST']
    class_names = list(load_dict['class_names'].values())
    total_class = len(class_names)
    all_class_names     = {}
    all_class_names_ori = {}
    for i in range(0, train_class_number):
        all_class_names['mscoco-train-'+str(i)] = class_names[i]
        all_class_names_ori['mscoco-train-'+str(i)] = class_names[i]

    for i in range(train_class_number, train_class_number+valid_class_number):
        all_class_names['mscoco-val-' + str(i)] = class_names[i]
        all_class_names_ori['mscoco-val-' + str(i)] = class_names[i]

    for i in range(train_class_number + valid_class_number, total_class):
        all_class_names['mscoco-test-' + str(i)] = class_names[i]
        all_class_names_ori['mscoco-test-' + str(i)] = class_names[i]


    return all_class_names_ori,all_class_names

def load_dtd_json(dataspec_json_pth):
    with open(dataspec_json_pth, 'r') as load_f:
        load_dict = json.load(load_f)

    train_class_number = load_dict['classes_per_split']['TRAIN']
    valid_class_number = load_dict['classes_per_split']['VALID']
    test_class_number = load_dict['classes_per_split']['TEST']
    class_names = list(load_dict['class_names'].values())
    total_class = len(class_names)
    all_class_names     = {}
    all_class_names_ori = {}
    for i in range(0, train_class_number):
        all_class_names['dtd-'+'train-'+str(i)] = class_names[i]
        all_class_names_ori['dtd-'+'train-'+str(i)] = class_names[i]

    for i in range(train_class_number, train_class_number+valid_class_number):
        all_class_names['dtd-'+'val-'+str(i)] = class_names[i]
        all_class_names_ori['dtd-'+'val-'+str(i)] = class_names[i]

    for i in range(train_class_number + valid_class_number, total_class):
        all_class_names['dtd-'+'test-'+str(i)] = class_names[i]
        all_class_names_ori['dtd-'+'test-'+str(i)] = class_names[i]

    return all_class_names_ori,all_class_names

def load_vgg_flower_json(dataspec_json_pth):
    with open(dataspec_json_pth, 'r') as load_f:
        load_dict = json.load(load_f)

    train_class_number = load_dict['classes_per_split']['TRAIN']
    valid_class_number = load_dict['classes_per_split']['VALID']
    test_class_number = load_dict['classes_per_split']['TEST']
    class_names = list(load_dict['class_names'].values())
    total_class = len(class_names)
    train_class_names = []
    val_class_names   = []
    test_class_names = []
    all_class_names_ori = {}
    for i in range(0, train_class_number):
        all_class_names_ori['vgg_flower-train-'+str(i)] = class_names[i]
        train_class_names.append(class_names[i])
    for i in range(train_class_number, train_class_number+valid_class_number):
        all_class_names_ori['vgg_flower-val-'+str(i)] = class_names[i]
        val_class_names.append(class_names[i])
    for i in range(train_class_number + valid_class_number, total_class):
        test_class_names.append(class_names[i])
        all_class_names_ori['vgg_flower-test-'+str(i)] = class_names[i]

    all_class_names = {}
    for i, name in enumerate(train_class_names):
        all_class_names['vgg_flower-train-'+str(i)] = name[4:]
    for i, name in enumerate(val_class_names):
        all_class_names['vgg_flower-val-'+str(i+train_class_number)] = name[4:]
    for i, name in enumerate(test_class_names):
        all_class_names['vgg_flower-test-'+str(i+train_class_number+valid_class_number)]= name[4:]



    return all_class_names_ori, all_class_names


def load_omniglot_json(dataspec_json_pth):
    with open(dataspec_json_pth, 'r') as load_f:
        load_dict = json.load(load_f)

    train_superclass_number = load_dict['superclasses_per_split']['TRAIN']
    valid_superclass_number = load_dict['superclasses_per_split']['VALID']
    test_superclass_number = load_dict['superclasses_per_split']['TEST']

    classes_per_superclass = list(load_dict['classes_per_superclass'].values())

    train_class_number = 0
    valid_class_number = 0
    test_class_number = 0

    for i in range(0, train_superclass_number):
        train_class_number += classes_per_superclass[i]

    for i in range(train_superclass_number, train_superclass_number + valid_superclass_number):
        valid_class_number += classes_per_superclass[i]

    for i in range(train_superclass_number + valid_superclass_number,
                   train_superclass_number + valid_superclass_number + test_superclass_number):
        test_class_number += classes_per_superclass[i]

    class_names = list(load_dict['class_names'].values())
    total_class = len(class_names)
    train_class_names = []
    val_class_names = []
    test_class_names = []
    all_class_names_ori = {}
    for i in range(0, train_class_number):
        all_class_names_ori['omniglot-train-' + str(i)] = class_names[i]
        train_class_names.append(class_names[i])
    for i in range(train_class_number, train_class_number + valid_class_number):
        all_class_names_ori['omniglot-train-' + str(i)] = class_names[i]
        val_class_names.append(class_names[i])
    for i in range(train_class_number + valid_class_number, total_class):
        all_class_names_ori['omniglot-train-' + str(i)] = class_names[i]
        test_class_names.append(class_names[i])
    all_class_names = {}
    for j, item in enumerate(train_class_names):
        item_split = item.split('_')
        new_name = ''
        if len(item_split) == 1:
            new_name = item_split[0]
            all_class_names['omniglot-train-' + str(j)] = new_name
        else:
            for i in range(len(item_split)):
                new_name = new_name + item_split[i]
                if i == (len(item_split) - 1):
                    break
                else:
                    new_name = new_name + ' '
            all_class_names['omniglot-train-' + str(j)] = new_name

    for j, item in enumerate(val_class_names):
        item_split = item.split('_')
        new_name = ''
        if len(item_split) == 1:
            new_name = item_split[0]
            all_class_names['omniglot-val-' + str(j + train_class_number)] = new_name
        else:
            for i in range(len(item_split)):
                new_name = new_name + item_split[i]
                if i == (len(item_split) - 1):
                    break
                else:
                    new_name = new_name + ' '
            all_class_names['omniglot-val-' + str(j + train_class_number)] = new_name

    for j, item in enumerate(test_class_names):
        item_split = item.split('_')
        new_name = ''
        if len(item_split) == 1:
            new_name = item_split[0]
            all_class_names['omniglot-test-' + str(j + train_class_number + valid_class_number)] = new_name
        else:
            for i in range(len(item_split)):
                new_name = new_name + item_split[i]
                if i == (len(item_split) - 1):
                    break
                else:
                    new_name = new_name + ' '
            all_class_names['omniglot-test-' + str(j + train_class_number + valid_class_number)] = new_name

    return all_class_names_ori, all_class_names

def load_quickdraw_json(dataspec_json_pth):
    with open(dataspec_json_pth, 'r') as load_f:
        load_dict = json.load(load_f)

    train_class_number = load_dict['classes_per_split']['TRAIN']
    valid_class_number = load_dict['classes_per_split']['VALID']
    test_class_number = load_dict['classes_per_split']['TEST']
    class_names = list(load_dict['class_names'].values())
    total_class = len(class_names)
    all_class_names_ori = {}
    all_class_names  = {}
    for i in range(0, train_class_number):

        all_class_names_ori['quickdraw-train-'+str(i)] = class_names[i]
        all_class_names['quickdraw-train-'+str(i)]     = class_names[i]
    for i in range(train_class_number, train_class_number+valid_class_number):

        all_class_names_ori['quickdraw-val-'+str(i)] = class_names[i]
        all_class_names['quickdraw-val-'+str(i)]     = class_names[i]
    for i in range(train_class_number + valid_class_number, total_class):

        all_class_names_ori['quickdraw-test-'+str(i)] = class_names[i]
        all_class_names['quickdraw-test-'+str(i)]     = class_names[i]

    return all_class_names_ori,all_class_names

def load_fungi_json(dataspec_json_pth):
    with open(dataspec_json_pth, 'r') as load_f:
        load_dict = json.load(load_f)

    train_class_number = load_dict['classes_per_split']['TRAIN']
    valid_class_number = load_dict['classes_per_split']['VALID']
    test_class_number = load_dict['classes_per_split']['TEST']

    class_names = list(load_dict['class_names'].values())
    total_class = len(class_names)
    train_class_names = []
    val_class_names   = []
    test_class_names = []
    all_class_names_ori = {}
    for i in range(0, train_class_number):
        all_class_names_ori['fungi-train-'+str(i)] = class_names[i]
        train_class_names.append(class_names[i])
    for i in range(train_class_number, train_class_number+valid_class_number):
        all_class_names_ori['fungi-val-'+str(i)] = class_names[i]

        val_class_names.append(class_names[i])
    for i in range(train_class_number + valid_class_number, total_class):
        all_class_names_ori['fungi-test-'+str(i)] = class_names[i]
        test_class_names.append(class_names[i])


    all_class_names = {}

    for i,name in enumerate(train_class_names):
        all_class_names['fungi-train-'+str(i)] = name[5:]
    for i,name in enumerate(val_class_names):
        all_class_names['fungi-val-'+str(i+train_class_number)] = name[5:]

    for i, name in enumerate(test_class_names):
        all_class_names['fungi-test-'+str(i+valid_class_number+train_class_number)] = name[5:]

    return all_class_names_ori,all_class_names

def load_traffic_sign_json(dataspec_json_pth):
    with open(dataspec_json_pth, 'r') as load_f:
        load_dict = json.load(load_f)

    train_class_number = load_dict['classes_per_split']['TRAIN']
    valid_class_number = load_dict['classes_per_split']['VALID']
    class_names        = list(load_dict['class_names'].values())

    total_class       = len(class_names)
    train_class_names = []
    val_class_names   = []
    test_class_names  = []
    all_class_names_ori  = {}
    for i in range(0, train_class_number):
        all_class_names_ori['traffic_sign-train-'+str(i)] = class_names[i]
        train_class_names.append(class_names[i])
    for i in range(train_class_number, train_class_number+valid_class_number):
        all_class_names_ori['traffic_sign-val-'+str(i)] = class_names[i]
        val_class_names.append(class_names[i])
    for i in range(train_class_number + valid_class_number, total_class):
        all_class_names_ori['traffic_sign-test-'+str(i)] = class_names[i]
        test_class_names.append(class_names[i])

    all_class_names = {}

    for i,name in enumerate(train_class_names):
        all_class_names['traffic_sign-train-'+str(i)] = name[3:]

    for i, name in enumerate(val_class_names):
        all_class_names['traffic_sign-val-'+str(i+train_class_number)] = name[3:]

    for i,name in enumerate(test_class_names):
        all_class_names['traffic_sign-test-'+str(i+train_class_number+valid_class_number)] = name[3:]

    return all_class_names_ori,all_class_names


def load_aircraft_json(dataspec_json_pth):
    with open(dataspec_json_pth, 'r') as load_f:
        load_dict = json.load(load_f)

    train_class_number = load_dict['classes_per_split']['TRAIN']
    valid_class_number = load_dict['classes_per_split']['VALID']
    class_names        = list(load_dict['class_names'].values())

    total_class       = len(class_names)
    all_class_names_ori = {}
    all_class_names     = {}
    for i in range(0, train_class_number):
        all_class_names_ori['aircraft-train-'+str(i)] = class_names[i]
        all_class_names['aircraft-train-'+str(i)] = class_names[i]

    for i in range(train_class_number, train_class_number+valid_class_number):
        all_class_names_ori['aircraft-val-'+str(i)] = class_names[i]
        all_class_names['aircraft-val-'+str(i)] = class_names[i]


    for i in range(train_class_number + valid_class_number, total_class):
        all_class_names_ori['aircraft-test-'+str(i)] = class_names[i]
        all_class_names['aircraft-test-'+str(i)] = class_names[i]


    return all_class_names_ori,all_class_names

def load_mnist_json(dataspec_json_pth):
    with open(dataspec_json_pth, 'r') as load_f:
        load_dict = json.load(load_f)

    train_class_number = load_dict['classes_per_split']['TRAIN']
    valid_class_number = load_dict['classes_per_split']['VALID']
    class_names        = list(load_dict['class_names'].values())

    total_class       = len(class_names)
    all_class_names_ori = {}
    all_class_names     = {}
    for i in range(0, train_class_number):
        class_name = class_names[i]
        class_name = class_name.split('_')
        class_name = 'number '+ class_name[-1]
        all_class_names_ori['mnist-train-'+str(i)] = class_names[i]
        all_class_names['mnist-train-'+str(i)] = class_name

    for i in range(train_class_number, train_class_number+valid_class_number):
        class_name = class_names[i]
        class_name = class_name.split('_')
        class_name = 'number ' + class_name[-1]

        all_class_names_ori['mnist-val-'+str(i)] = class_names[i]
        all_class_names['mnist-val-'+str(i)] = class_name


    for i in range(train_class_number + valid_class_number, total_class):
        class_name = class_names[i]
        class_name = class_name.split('_')
        class_name = 'number ' + class_name[-1]
        all_class_names_ori['mnist-test-'+str(i)] = class_names[i]
        all_class_names['mnist-test-'+str(i)] = class_name


    return all_class_names_ori,all_class_names


def load_cifar10_json(dataspec_json_pth):
    with open(dataspec_json_pth, 'r') as load_f:
        load_dict = json.load(load_f)

    train_class_number = load_dict['classes_per_split']['TRAIN']
    valid_class_number = load_dict['classes_per_split']['VALID']
    class_names        = list(load_dict['class_names'].values())

    total_class       = len(class_names)
    all_class_names_ori = {}
    all_class_names     = {}
    for i in range(0, train_class_number):
        all_class_names_ori['cifar10-train-'+str(i)] = class_names[i]
        all_class_names['cifar10-train-'+str(i)] = class_names[i]

    for i in range(train_class_number, train_class_number+valid_class_number):
        all_class_names_ori['cifar10-val-'+str(i)] = class_names[i]
        all_class_names['cifar10-val-'+str(i)] = class_names[i]


    for i in range(train_class_number + valid_class_number, total_class):
        all_class_names_ori['cifar10-test-'+str(i)] = class_names[i]
        all_class_names['cifar10-test-'+str(i)] = class_names[i]

    return all_class_names_ori,all_class_names


def load_cifar100_json(dataspec_json_pth):
    with open(dataspec_json_pth, 'r') as load_f:
        load_dict = json.load(load_f)

    train_class_number = load_dict['classes_per_split']['TRAIN']
    valid_class_number = load_dict['classes_per_split']['VALID']
    class_names        = list(load_dict['class_names'].values())

    total_class       = len(class_names)
    all_class_names_ori = {}
    all_class_names     = {}
    for i in range(0, train_class_number):
        all_class_names_ori['cifar100-train-'+str(i)] = class_names[i]
        all_class_names['cifar100-train-'+str(i)] = class_names[i]

    for i in range(train_class_number, train_class_number+valid_class_number):
        all_class_names_ori['cifar100-val-'+str(i)] = class_names[i]
        all_class_names['cifar100-val-'+str(i)] = class_names[i]


    for i in range(train_class_number + valid_class_number, total_class):
        all_class_names_ori['cifar100-test-'+str(i)] = class_names[i]
        all_class_names['cifar100-test-'+str(i)] = class_names[i]

    return all_class_names_ori,all_class_names

def get_class_names(dset_name,dset_pth):
    if dset_name == 'cu_birds':
        return load_cu_birds_json(dset_pth)

    if dset_name == 'vgg_flower':
        return load_vgg_flower_json(dset_pth)

    if dset_name == 'dtd':
        return load_dtd_json(dset_pth)

    if dset_name == 'quickdraw':
        return load_quickdraw_json(dset_pth)

    if dset_name == 'mscoco':
        return load_mscoco_json(dset_pth)

    if dset_name == 'ilsvrc':
        return load_ilsvrc_json(dset_pth)

    if dset_name == 'fungi':
        return load_fungi_json(dset_pth)

    if dset_name == 'traffic_sign':
        return load_traffic_sign_json(dset_pth)

    if dset_name == 'omniglot':
        return load_omniglot_json(dset_pth)

    if dset_name == 'aircraft':
        return load_aircraft_json(dset_pth)

    if dset_name == 'mnist':
        return load_mnist_json(dset_pth)

    if dset_name == 'cifar10':
        return load_cifar10_json(dset_pth)

    if dset_name == 'cifar100':
        return load_cifar100_json(dset_pth)


if __name__ == '__main__':

    METADATASET_NAMES = ['cu_birds', 'dtd', 'mscoco', 'vgg_flower', 'traffic_sign', 'quickdraw',
                         'fungi']
    for dataset in METADATASET_NAMES:
        print(dataset)
        dset_pth = f'./meta_dataset/dataset_conversion/dataset_specs/{dataset}_dataset_spec.json'
        tr,ts,_ = get_class_names(dataset,dset_pth,'bert')
        print(tr)

