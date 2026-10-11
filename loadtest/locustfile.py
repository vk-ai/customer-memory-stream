import random
from locust import HttpUser, constant, task

QS = ["Which city should the courier deliver to?", "What is the order number we are discussing?",
      "How does the customer prefer to be contacted?", "Which subscription plan is the customer on?",
      "Any dietary restrictions for the meal kit?", "Which device is the app running on?"]


class Agent(HttpUser):
    wait_time = constant(0)

    @task(9)
    def search(self):
        self.client.get(f"/customers/cust-{random.randrange(200):04d}/memory",
                        params={"q": random.choice(QS), "k": 5}, name="/customers/[id]/memory")

    @task(1)
    def ingest(self):
        n = random.randrange(10**9)
        self.client.post("/events", json={"event_id": f"lt-{n}", "customer_id": f"cust-{n % 200:04d}",
                                          "text": "Load test chatter message", "ts": 1.8e9}, name="/events")
