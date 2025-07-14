# noxfile.py
import nox

@nox.session
def tests(session):
    session.install("pytest", "numpy", "scipy", "soundfile", "sounddevice", "cryptography")
    session.run("pytest")

@nox.session
def lint(session):
    session.install("black", "flake8", "mypy", "numpy", "scipy")
    session.run("black", "--check", "rtwm", "tests")
    session.run("flake8", "rtwm", "tests")
    session.run("mypy", "rtwm")

@nox.session
def format(session):
    session.install("black")
    session.run("black", "rtwm", "tests")
