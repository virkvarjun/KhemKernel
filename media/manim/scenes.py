"""3Blue1Brown-style animations for the KhemKernel project.

Black background, Computer Modern (CMU Serif / CMU Typewriter), the manim
default palette (which is 3b1b's). No LaTeX: math is set with the real Computer
Modern font through Pango, with fractions and sub/superscripts hand-composed.

Render one at a time, e.g.:
    manim -qh --format mp4 scenes.py Pipeline -o 01_pipeline
"""
from manim import *

config.background_color = "#000000"

SERIF = "CMU Serif"
MONO = "CMU Typewriter Text"

# 3b1b-ish role colors
C_ATOM = BLUE_C
C_NAME = YELLOW
C_PROC = TEAL
C_OK = GREEN
C_HOT = "#FC6255"  # manim RED_C / 3b1b red


def serif(s, size=40, color=WHITE, weight=NORMAL):
    return Text(s, font=SERIF, font_size=size, color=color, weight=weight)


def mono(s, size=36, color=WHITE):
    return Text(s, font=MONO, font_size=size, color=color)


def mk(s, size=44, color=WHITE):
    return MarkupText(s, font=SERIF, font_size=size, color=color)


def frac(num, den, color=WHITE):
    w = max(num.width, den.width) + 0.15
    bar = Line(LEFT, RIGHT, color=color).set_width(w)
    num.next_to(bar, UP, buff=0.1)
    den.next_to(bar, DOWN, buff=0.1)
    return VGroup(num, bar, den)


def token_box(s, color=WHITE, fill=BLACK, size=34):
    t = mono(s, size=size, color=color)
    box = SurroundingRectangle(t, color=color, buff=0.13, corner_radius=0.06)
    box.set_fill(fill, opacity=0.12)
    return VGroup(box, t)


class Pipeline(Scene):
    """SMILES -> tokens -> transformer -> IUPAC name. ~9s"""
    def construct(self):
        title = serif("From a molecule to its name", size=44).to_edge(UP)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=1.2)

        smiles = mono("c1ccc(O)cc1", size=42, color=C_ATOM).shift(LEFT * 4.2)
        self.play(Write(smiles), run_time=1.2)

        # tokenize into a few colored boxes
        toks = ["c1", "c", "c", "c", "(O)", "c", "c1"]
        cols = [C_ATOM, C_ATOM, C_ATOM, C_ATOM, C_HOT, C_ATOM, C_ATOM]
        boxes = VGroup(*[token_box(t, color=c, size=26) for t, c in zip(toks, cols)])
        boxes.arrange(RIGHT, buff=0.12).shift(LEFT * 4.2)
        self.play(ReplacementTransform(smiles, boxes), run_time=1.2)

        # transformer block
        block = RoundedRectangle(width=2.6, height=1.5, corner_radius=0.12, color=C_PROC)
        block.set_fill(C_PROC, opacity=0.10)
        label = serif("transformer", size=28, color=C_PROC).move_to(block)
        tf = VGroup(block, label)
        a1 = Arrow(boxes.get_right(), block.get_left(), buff=0.25, color=GREY_B)
        self.play(GrowArrow(a1), FadeIn(tf, scale=0.9), run_time=1.0)

        name = serif("phenol", size=52, color=C_NAME).shift(RIGHT * 4.3)
        a2 = Arrow(block.get_right(), name.get_left(), buff=0.25, color=GREY_B)
        self.play(GrowArrow(a2), run_time=0.6)
        self.play(Write(name), run_time=1.0)
        self.play(Indicate(name, color=C_NAME, scale_factor=1.15), run_time=1.0)
        self.wait(0.6)


class BPE(Scene):
    """Byte pair merges: characters collapse into a morpheme token. ~9s"""
    def construct(self):
        title = serif("Byte pair encoding learns morphemes", size=40).to_edge(UP)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=1.0)

        chars = list("benzoic")
        boxes = VGroup(*[token_box(c, color=WHITE, size=34) for c in chars])
        boxes.arrange(RIGHT, buff=0.14)
        self.play(LaggedStart(*[FadeIn(b, scale=0.8) for b in boxes], lag_ratio=0.12), run_time=1.4)

        def merge(group, i, text, run_time=1.0):
            # highlight the adjacent pair i, i+1 then fuse into one box
            a, b = group[i], group[i + 1]
            self.play(a[0].animate.set_color(C_HOT), b[0].animate.set_color(C_HOT), run_time=0.5)
            new = token_box(text, color=C_NAME, size=34)
            new.move_to(VGroup(a, b))
            self.play(ReplacementTransform(VGroup(a, b), new), run_time=run_time)
            new_group = VGroup(*group[:i], new, *group[i + 2:])
            new_group.arrange(RIGHT, buff=0.14)
            self.play(new_group.animate, run_time=0.4)
            return new_group

        boxes = merge(boxes, 0, "be")
        boxes = merge(boxes, 0, "ben")
        boxes = merge(boxes, 0, "benz")
        # collapse the rest in one sweep
        whole = token_box("benzoic", color=C_NAME, size=40).move_to(boxes)
        self.play(ReplacementTransform(boxes, whole), run_time=1.0)
        cap = mono("benzoic acid  ->  1 token", size=28, color=GREY_B).next_to(whole, DOWN, buff=0.7)
        self.play(FadeIn(cap), run_time=0.8)
        self.wait(0.6)


class Attention(Scene):
    """The attention equation + a softmax row lighting up. ~9s"""
    def construct(self):
        title = serif("Attention", size=46).to_edge(UP)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.9)

        # equation: Attention(Q,K,V) = softmax( QKᵀ / √dₖ ) V
        lhs = mk("Attention(Q, K, V) =", size=44)
        sm_l = mk("softmax", size=44)
        num = mk("QK<sup>T</sup>", size=40, color=C_ATOM)
        den = mk("√d<sub>k</sub>", size=40, color=C_HOT)
        fr = frac(num, den)
        lpar = mk("(", size=56)
        rpar = mk(")", size=56)
        v = mk("V", size=44, color=C_NAME)
        eq = VGroup(lhs, sm_l, lpar, fr, rpar, v).arrange(RIGHT, buff=0.18)
        eq.move_to(UP * 1.2).scale(0.95)
        self.play(Write(eq), run_time=2.2)

        # a softmax row: 7 cells, weights light up
        cells = VGroup(*[Square(0.7, color=GREY_B) for _ in range(7)])
        cells.arrange(RIGHT, buff=0.12).shift(DOWN * 1.4)
        self.play(Create(cells), run_time=0.8)
        weights = [0.05, 0.08, 0.55, 0.12, 0.05, 0.10, 0.05]
        self.play(*[c.animate.set_fill(C_ATOM, opacity=w * 1.6) for c, w in zip(cells, weights)], run_time=1.2)
        q = serif("query attends here", size=26, color=C_ATOM).next_to(cells[2], DOWN, buff=0.3)
        arr = Arrow(q.get_top(), cells[2].get_bottom(), buff=0.1, color=C_ATOM)
        self.play(GrowArrow(arr), FadeIn(q), run_time=1.0)
        self.wait(0.7)


class Verifier(Scene):
    """name -> OPSIN -> molecule -> match -> the accuracy jump. ~9s"""
    def construct(self):
        title = serif("A verifier you get for free", size=42).to_edge(UP)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=1.0)

        name = serif("phenol", size=46, color=C_NAME).shift(LEFT * 4.5 + UP * 0.5)
        self.play(Write(name), run_time=0.9)

        opsin = RoundedRectangle(width=1.7, height=0.9, corner_radius=0.1, color=C_PROC)
        opsin.set_fill(C_PROC, opacity=0.1).shift(LEFT * 1.6 + UP * 0.5)
        olab = serif("OPSIN", size=26, color=C_PROC).move_to(opsin)
        a1 = Arrow(name.get_right(), opsin.get_left(), buff=0.2, color=GREY_B)
        self.play(GrowArrow(a1), FadeIn(VGroup(opsin, olab)), run_time=0.9)

        # benzene + OH glyph
        ring = RegularPolygon(6, radius=0.55, color=C_ATOM).shift(RIGHT * 1.4 + UP * 0.5)
        oh = mono("OH", size=26, color=C_HOT).next_to(ring, UR, buff=0.05)
        mol = VGroup(ring, oh)
        a2 = Arrow(opsin.get_right(), ring.get_left(), buff=0.2, color=GREY_B)
        self.play(GrowArrow(a2), Create(ring), FadeIn(oh), run_time=1.0)

        check = serif("✓ same molecule", size=34, color=C_OK).shift(RIGHT * 4.4 + UP * 0.5)
        self.play(Write(check), run_time=0.9)

        # accuracy jump bar
        base = Line(LEFT * 4, RIGHT * 4, color=GREY_D).shift(DOWN * 2.2)
        g = serif("greedy  79.5%", size=28, color=GREY_B).next_to(base, UP, buff=0.2).align_to(base, LEFT)
        self.play(FadeIn(base), FadeIn(g), run_time=0.6)
        bar = Rectangle(width=0.05, height=0.5, color=C_OK, fill_opacity=1).next_to(base, UP, buff=0.05).align_to(base, LEFT)
        self.play(bar.animate.stretch_to_fit_width(8 * 0.958).align_to(base, LEFT), run_time=1.2)
        v = serif("verified  95.8%", size=30, color=C_OK).next_to(base, DOWN, buff=0.3)
        self.play(FadeIn(v, shift=UP * 0.2), run_time=0.8)
        self.wait(0.6)


class GradCheck(Scene):
    """built from scratch: the finite-difference check. ~8s"""
    def construct(self):
        title = serif("Every gradient, by hand", size=44).to_edge(UP)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=1.0)

        # central difference: dL/dx ≈ (L(x+ε) - L(x-ε)) / (2ε)
        lhs = mk("∂L / ∂x<sub>i</sub>  ≈", size=46)
        num = mk("L(x+ε) − L(x−ε)", size=40, color=C_ATOM)
        den = mk("2ε", size=40, color=C_HOT)
        fr = frac(num, den)
        eq = VGroup(lhs, fr).arrange(RIGHT, buff=0.25).move_to(UP * 0.8)
        self.play(Write(eq), run_time=2.0)

        analytic = serif("analytic", size=34, color=C_NAME)
        approx = serif("≈", size=40)
        numeric = serif("numeric", size=34, color=C_ATOM)
        row = VGroup(analytic, approx, numeric).arrange(RIGHT, buff=0.4).shift(DOWN * 0.8)
        self.play(FadeIn(analytic, shift=RIGHT * 0.2), run_time=0.5)
        self.play(FadeIn(numeric, shift=LEFT * 0.2), run_time=0.5)
        self.play(Write(approx), run_time=0.5)

        err = mono("max error  <  3 x 10^-9", size=30, color=C_OK).shift(DOWN * 2.0)
        chk = serif("✓", size=40, color=C_OK).next_to(err, RIGHT, buff=0.3)
        self.play(Write(err), run_time=1.0)
        self.play(FadeIn(chk, scale=1.4), run_time=0.6)
        self.wait(0.6)


class TiledMatmul(Scene):
    """tiles stream from global into shared memory and accumulate. ~9s"""
    def construct(self):
        title = serif("The tiled matmul", size=44).to_edge(UP)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=1.0)

        eq = mk("C = A · B", size=44).next_to(title, DOWN, buff=0.3)
        self.play(Write(eq), run_time=0.8)

        # global memory band
        gm = RoundedRectangle(width=10, height=1.2, corner_radius=0.1, color=GREY_D).shift(UP * 0.6)
        gml = mono("global memory", size=22, color=GREY_B).move_to(gm.get_corner(UL) + RIGHT * 1.0 + DOWN * 0.25)
        self.play(Create(gm), FadeIn(gml), run_time=0.7)

        tA = VGroup(*[Square(0.32, color=C_ATOM, fill_opacity=0.5) for _ in range(4)]).arrange(RIGHT, buff=0.04)
        tB = VGroup(*[Square(0.32, color=C_NAME, fill_opacity=0.5) for _ in range(4)]).arrange(RIGHT, buff=0.04)
        tA.move_to(gm).shift(LEFT * 2.5)
        tB.move_to(gm).shift(RIGHT * 2.5)
        self.play(FadeIn(tA), FadeIn(tB), run_time=0.6)

        # shared memory box
        sm = RoundedRectangle(width=4.4, height=1.5, corner_radius=0.1, color=C_PROC).shift(DOWN * 1.4)
        sml = mono("shared memory", size=22, color=C_PROC).next_to(sm, UP, buff=0.1)
        self.play(Create(sm), FadeIn(sml), run_time=0.7)

        sA = tA.copy(); sB = tB.copy()
        self.play(sA.animate.move_to(sm).shift(LEFT * 1.0), sB.animate.move_to(sm).shift(RIGHT * 1.0), run_time=1.2)
        self.play(Indicate(VGroup(sA, sB), color=C_PROC, scale_factor=1.1), run_time=0.8)

        # accumulate into one output cell
        out = Square(0.7, color=C_OK).shift(DOWN * 1.4 + RIGHT * 4.5)
        acc = mono("acc += sA·sB", size=24, color=C_OK).next_to(out, UP, buff=0.15)
        self.play(Create(out), FadeIn(acc), run_time=0.6)
        self.play(out.animate.set_fill(C_OK, opacity=0.9), run_time=1.0)
        self.wait(0.6)
