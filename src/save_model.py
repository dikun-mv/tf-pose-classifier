import sys
from model import get_model

if __name__ == '__main__':
    nb_class = int(sys.argv[1])
    w_path = sys.argv[2]
    m_path = sys.argv[3]

    if not nb_class or not w_path or not m_path:
        print('Unspecified args')
        exit(1)

    model = get_model(nb_class)
    model.load_weights(w_path)
    model.save(m_path)
