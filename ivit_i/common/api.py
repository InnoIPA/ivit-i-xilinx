import logging, sys, os
sys.path.append(os.path.abspath(".."))

TAG_KEY = "TAG"

def get(prim_conf):
    """ Get target API and return it """
    # model_conf = prim_conf[ prim_conf['framework'] ]    

    if 'cls' in prim_conf[TAG_KEY]:
        from ivit_i.cls.classification import Classification
        trg = Classification(prim_conf)
    elif 'obj' in prim_conf[TAG_KEY]:
        from ivit_i.obj.yolov3 import YOLOv3
        trg = YOLOv3(prim_conf)
    else:
        msg = 'Unexcepted `tag` in {}'.format(prim_conf['prim'])
        logging.error(msg)
        raise Exception(msg)

    return trg