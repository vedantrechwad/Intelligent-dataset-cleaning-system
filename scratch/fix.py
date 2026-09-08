with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix the NoneType issue first
content = content.replace(
    'qscore = st.session_state.raw_quality_score',
    'qscore = st.session_state.raw_quality_score or {}'
)

# Replace use_container_width
# We must only replace in occurrences where it's used as a kwarg.
# For simplicity:
content = content.replace('use_container_width=True', 'width="stretch"')
content = content.replace('use_container_width=False', 'width="content"')

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Replacement complete.")
