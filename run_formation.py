from environments import FormationEnv

def main():
    # instantiate the gym environment

    form = FormationEnv()

    while not form.done:
        action = None
        form.step(None)
        form.render()


if __name__ == "__main__":
    main()
