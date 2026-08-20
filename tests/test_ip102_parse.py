"""Verify the real IP102 dataset parses correctly.

Checks: 0 missing images on disk, image/label counts match the JSON
annotations, class count is as expected. Run:

    python tests/test_ip102_parse.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))

from libml.ip102_data import find_data_root  # noqa: E402
from libml.ip102_data import get_data_manager  # noqa: E402

EXPECTED_CLASSES = 25


def main():
  root = find_data_root()
  print('data_root :', root)
  dm = get_data_manager('ip102', data_root=root, shuffle=True, seed=1993,
                        split_val=True)
  report = dm.verify()
  print('image_dir :', report['image_dir'])
  print('num_classes:', report['num_classes'], '| val_split:',
        report['val_split'])
  print('class_ids :', report['class_ids'])

  assert report['num_classes'] == EXPECTED_CLASSES, \
      'Expected %d classes, got %d' % (EXPECTED_CLASSES,
                                       report['num_classes'])

  for split in ('train', 'val', 'test'):
    assert split in report, 'Missing split %s' % split
    r = report[split]
    print('  %-5s images=%d anns=%d labeled=%d missing=%d'
          % (split, r['images_in_json'], r['annotations'],
             r['images_labeled'], r['missing_images']))
    assert r['missing_images'] == 0, \
        '%s has %d missing images' % (split, r['missing_images'])
    assert r['images_labeled'] == r['images_in_json'], \
        '%s image/label count mismatch' % split
    assert r['images_in_json'] > 0, '%s is empty' % split

  # per-split per-class counts (sanity: every class present in train)
  for c, count in enumerate(dm.split_counts('train')):
    assert count > 0, 'train class %d has no images' % c

  total_train = sum(dm.split_counts('train'))
  assert total_train == report['train']['images_labeled']
  print('train total images:', total_train)

  # task sizes must match the 7/6/6/6 specification
  assert dm.task_sizes == [7, 6, 6, 6], dm.task_sizes
  print('task_sizes :', dm.task_sizes)
  print('\nALL DATASET PARSE CHECKS PASSED (0 missing, counts match)')


if __name__ == '__main__':
  main()