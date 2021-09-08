from environments import FormationEnv_jmc

def main():
    # instantiate the gym environment

    form = FormationEnv_jmc()

    while not form.done:
        action = None
        form.step(None)
        form.render()


if __name__ == "__main__":
    main()
