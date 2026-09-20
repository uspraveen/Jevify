"""jev-bench: real, human-labeled data reformatted into System One questions.

Every record is one (state, question, label) triple in the exact wire format a
System One model consumes, plus a ground-truth label and, where the source
provides one, a human label *distribution* (calibration gold).
"""
