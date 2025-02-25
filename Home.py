import streamlit as st
import time

logo_url = 'https://www.kdan.com/blog/wp-content/uploads/2024/06/KDAN_blog_Nav-Bar%E7%94%A8%E5%9C%96%EF%BC%88h80%EF%BC%89.png'
st.logo(logo_url)
st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)


st.sidebar.title("Navigation")
st.sidebar.markdown(
    """
    - [Visit our Notion page](https://adnex-data.notion.site/ADNEX-6f4ca3d1731744a3bf0058943abd2e61?pvs=4)
    - [Best Practices for Team Development](https://adnex-data.notion.site/a4cb7818869743e083f45bbd2b16f699)
    """
)

markdown_content = """
# Welcome to ADNEX Playground! 👋

We using Streamlit as our team playground
Streamlit is an open-source app framework built specifically for
Machine Learning and Data Science projects.

**👈 Select a project from the sidebar** to see some examples
of what we have done!

## About Us
We are the ADNEX team, dedicated to leveraging data and technology to drive innovation and excellence in our projects.

## Want to learn more?
- Check out [streamlit.io](https://streamlit.io)
- Jump into our [documentation](https://docs.streamlit.io)
- Ask a question in our [community forums](https://discuss.streamlit.io)


## Technical Questions?
- **Bennett** - Analytics Engineer
- **Eric** - Data Engineer


## Here’s how to reach us:

- Skype channel: https://join.skype.com/uh6Pnhqb3uuE
- Mattermost channel: https://chat.kdan.cc/kdan/channels/intern-adnex_data
- Email: adcloud@kdanmobile.com
- Team manager: Robert

---

We are committed to making data accessible and actionable for everyone. Thank you for visiting our playground!
"""

st.markdown(markdown_content)

# # Show a spinner during a process
# with st.spinner(text='In progress'):
#     time.sleep(3)
#     st.success('Done')

# # Show and update progress bar
# bar = st.progress(50)
# time.sleep(3)
# bar.progress(100)

st.balloons()

# st.toast('ADNEX')
# st.error('Error message')
# st.warning('Warning message')
# st.info('Info message')
# st.success('Success message')
