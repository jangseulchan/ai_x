'''
samplec_pac/ab/__init__.py
from sample_pac.ab import b * 수행할 경우
    자동 import 될 모듈지정할 수 있음 (__all__)
samplec_pac/ab/a.py
samplec_pac/ab/b.py
'''
__all__ = ['a']
print('sample_pac패키지 안의 ab패키지 로드됩니다')