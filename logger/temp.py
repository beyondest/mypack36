import logging

fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

h = logging.FileHandler('temp.log','w','utf-8')
h.setLevel(logging.INFO)
h.setFormatter(fmt)



h2 = logging.StreamHandler()
h2.setLevel(logging.WARNING)
h2.setFormatter(fmt)
lr= logging.getLogger('camera')


lr.addHandler(h)
lr.addHandler(h2)
lr.setLevel(logging.CRITICAL)


lr.info('info')
lr.debug('dbug')
lr.warning('war')
lr.critical(f'fuck{4}')

