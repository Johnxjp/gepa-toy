---
license: cc-by-nc-4.0
pretty_name: Customer Support Tickets
size_categories:
- 10K<n<100K
task_categories:
- text-classification
- summarization
- question-answering
language:
- en
- de
tags:
- ticketsystem
- helpdesk
- customer
- support
- ticket
- routing
---

# Featuring Labeled Customer Emails and Support Responses

## 🔧 Synthetic IT Ticket Generator — **Custom Dataset**

Create a **dataset tailored to your own queues & priorities** (no PII).

👉 **[Generate custom data](https://open-ticket-ai.com/en/products/synthetic-data/synthetic-data-generation?utm_source=kaggle&utm_medium=readme&utm_campaign=sdg&utm_content=top)**

* Define **your queues, priorities, language**

*Need an on-prem AI to auto-classify tickets?*  
→ **[Open Ticket AI](https://openticketai.com/?utm_source=kaggle&utm_medium=readme&utm_campaign=otai&utm_content=secondary)**


There are 2 Versions of the dataset, the new version has more tickets, but only languages english and german. So please look at both files, to find what best fits your needs.
Checkout my Open Source Customer Support AI:
[Open Ticket AI](https://openticketai.com/en/?utm_source=kaggle&utm_medium=readme&utm_campaign=otai&utm_content=secondary)

Definetly check out my other Dataset:  
[Tickets from Github Issues](https://www.kaggle.com/datasets/tobiasbueck/helpdesk-github-tickets)

&gt;It includes **priorities**, **queues**, **types**, **tags**, and **business types**. This preview offers a detailed structure with classifications by department, type, priority, language, subject, full email text, and agent answers.

## Features / Attributes

| Field                        | Description                                                                        | Values                                                                                                               |
|------------------------------|------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| 🔀 **Queue**                 | Specifies the department to which the email ticket is routed                       | e.g. Technical Support, Customer Service, Billing and Payments, ...                                                  |
| 🚦 **Priority**              | Indicates the urgency and importance of the issue                                  | 🟢Low<br>🟠Medium<br>🔴Critical                                                                                        |
| 🗣️ **Language**             | Indicates the language in which the email is written                               | EN, DE                                                                                               |
| **Subject**                  | Subject of the customer's email                                                    |                                                                                                                      |
| **Body**                     | Body of the customer's email                                                       |                                                                                                                      |
| **Answer**                   | The response provided by the helpdesk agent                                        |                                                                                                                      |
| **Type**                     | The type of ticket as picked by the agent                                          | e.g. Incident, Request, Problem, Change ...                                                                          |
| 🏢 **Business Type**         | The business type of the support helpdesk                                          | e.g. Tech Online Store, IT Services, Software Development Company                                                    |
| **Tags**                     | Tags/categories assigned to the ticket, split into ten columns in the dataset        | e.g. "Software Bug", "Warranty Claim"                                                                                |

### Queue
Specifies the department to which the email ticket is categorized. This helps in routing the ticket to the appropriate support team for resolution.
- 💻 **Technical Support:** Technical issues and support requests.
- 🈂️ **Customer Service:** Customer inquiries and service requests.
- 💰 **Billing and Payments:** Billing issues and payment processing.
- 🖥️ **Product Support:** Support for product-related issues.
- 🌐 **IT Support:** Internal IT support and infrastructure issues.
- 🔄 **Returns and Exchanges:** Product returns and exchanges.
- 📞 **Sales and Pre-Sales:** Sales inquiries and pre-sales questions.
- 🧑‍💻 **Human Resources:** Employee inquiries and HR-related issues.
- ❌ **Service Outages and Maintenance:** Service interruptions and maintenance.
- 📮 **General Inquiry:** General inquiries and information requests.

### Priority
Indicates the urgency and importance of the issue. Helps in managing the workflow by prioritizing tickets that need immediate attention.
- 🟢 **1 (Low):** Non-urgent issues that do not require immediate attention. Examples: general inquiries, minor inconveniences, routine updates, and feature requests.
- 🟠 **2 (Medium):** Moderately urgent issues that need timely resolution but are not critical. Examples: performance issues, intermittent errors, and detailed user questions.
- 🔴 **3 (Critical):** Urgent issues that require immediate attention and quick resolution. Examples: system outages, security breaches, data loss, and major malfunctions.

### Language
Indicates the language in which the email is written. Useful for language-specific NLP models and multilingual support analysis.
- **en (English)**
- **de (German)**

### Answer
The response provided by the helpdesk agent, containing the resolution or further instructions. Useful for analyzing the quality and effectiveness of the support provided.

### Types
Different types of tickets categorized to understand the nature of the requests or issues.
- ❗ **Incident:** Unexpected issue requiring immediate attention.
- 📝 **Request:** Routine inquiry or service request.
- ⚠️ **Problem:** Underlying issue causing multiple incidents.
- 🔄 **Change:** Planned change or update.

### Tags
Tags/categories assigned to the ticket to further classify and identify common issues or topics.
- Examples: "Product Support," "Technical Support," "Sales Inquiry."

## Use Cases

| Task                          | Description                                                                                           |
|-------------------------------|-------------------------------------------------------------------------------------------------------|
| **Text Classification**       | Train machine learning models to accurately classify email content into appropriate departments, improving ticket routing and handling. |
| **Priority Prediction**       | Develop algorithms to predict the urgency of emails, ensuring that critical issues are addressed promptly. |
| **Customer Support Analysis** | Analyze the dataset to gain insights into common customer issues, optimize support processes, and enhance overall service quality. |

## Upvote this Dataset
Your support through an upvote would be greatly appreciated❤️🙂 Thank you.


## More Information

Other Datasets can also be found on Kaggle, including a Multi Language Ticket Dataset, which also has French, Spanish and Portuguese Tickets.


## Created By

[Softoft](https://softoft.de)