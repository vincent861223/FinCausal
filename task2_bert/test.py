import argparse
import collections
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from transformers import BertTokenizer
import os


def main(config, args):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        'data/train.csv',
        batch_size=2,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    with torch.no_grad():
        f = open(args.output, 'w')
        f.write('Index;Text;Cause;Effect\n')
        for i, batch in enumerate(tqdm(data_loader)):
            input_ids = batch['input_ids'].to(device)

            score = model(input_ids)
            pred = {}
            for key in score.keys():
                pred[key] = torch.max(score[key], dim=-1)[1]

            for j, idx in enumerate(batch['id']) :
                text = batch['text'][j]
                cause = batch['tokened_text'][j][pred['cause_start'][j]: pred['cause_end'][j]]
                cause = ' '.join(cause)
                #cause_start, cause_end = getCharPosition(text, cause)
                #cause = text[cause_start: cause_end]

                effect = batch['tokened_text'][j][pred['effect_start'][j]: pred['effect_end'][j]]
                effect = ' '.join(effect)
                #effect_start, effect_end = getCharPosition(text, effect)
                #effect = text[effect_start: effect_end]
                f.write('{};{};{};{}\n'.format(idx, text, cause, effect))

def getCharPosition(text, ref):
    if(len(ref) < 3 ): return -1, -1
    start = -1
    end = -1
    current = 0
    head = ref[:3]
    while current < len(text):
        pos = [text.find(substring, current) for substring in head]
        ok = True
        for a, b in zip(pos[:-1], pos[1:]): 
            ok &= (a <= b)
        if ok:
            start = pos[0]
            break
        elif -1 in pos:
            break
        else: 
            current = pos[0] + len(head[0])

    tail = ref[-3:]
    while current < len(text):
        pos = [text.find(substring, current) for substring in tail]
        ok = True
        for a, b in zip(pos[:-1], pos[1:]): 
            ok &= (a <= b)
        if ok:
            end = pos[-1] + len(tail[-1])
            break
        elif -1 in pos:
            break
        else: 
            current = pos[0] + len(tail[0])
    return start, end


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-o', '--output', default=None, type=str,
            help='indices of GPUs to enable (default: all)')
    

    config = ConfigParser.from_args(args)
    args = args.parse_args()
    main(config, args)
