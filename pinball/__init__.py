from . import Pinball


def run(field=None):
    """
    Shall we play a game?
    """
    with Pinball.Pinball() as p:
        p.setup(load_field=field)
        p.play()
