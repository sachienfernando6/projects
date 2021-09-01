import matplotlib.pyplot as p
from IPython import display

p.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(p.gcf())
    p.clf()
    p.title('Training...')
    p.xlabel('Number of Games')
    p.ylabel('Score')
    p.plot(scores)
    p.plot(mean_scores)
    p.ylim(ymin=0)
    p.text(len(scores)-1, scores[-1], str(scores[-1]))
    p.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    p.show(block=False)