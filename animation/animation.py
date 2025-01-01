from manim import *

class Intro(Scene):
    def construct(self):
        title = Text("RECONSTRUCT INCOMPLETE 3D MODELS USING SVD IN BLENDER")
        self.play(title, run_time=5)

class StepByStep(Scene):
    def construct(self):
        title = Text("Step-by-Step Process")
        self.play(Write(title, run_time=5))
        self.play(FadeOut(title))

        step1 = Text("Step 1: Data Preparation")
        self.play(Write(step1))
        self.play(step1.animate.shift(UP * 3))