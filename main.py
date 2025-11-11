from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
import os

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Define the State ---
class AppointmentState(TypedDict, total=False):
    request: str
    doctor_details: str
    appointment_details: str


# --- LLM Initialization ---
model = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=256
)

# --- Prompt Templates ---
doctor_selection_prompt = ChatPromptTemplate.from_template(
    """
    Context: {context}
    Select the best doctor based on the given user request: {request}
    Return only the doctor name and reason for selection.
    """
)

appointment_schedule_prompt = ChatPromptTemplate.from_template(
    """
    Appointment context (doctor availability): {appointment_context}
    Create an appointment schedule for the selected doctor: {doctor_details}
    """
)

# --- Chains (correct order of operators) ---
doctor_selection_chain = doctor_selection_prompt | model | StrOutputParser()
appointment_schedule_chain = appointment_schedule_prompt | model | StrOutputParser()

# --- Mock Data ---
context = {
    "Sourabh Singh": "Oncology Expert with 12 years of experience.",
    "Goutam Tak": "Dermatology Expert with 10 years of experience.",
    "Nikitha Vangale": "Cardiology Specialist with 20 years of experience."
}

appointment_context = {
    "Sourabh Singh": "Available Mon–Fri 10 AM–2 PM.",
    "Goutam Tak": "Available Wed–Sat 4 PM–6 PM.",
    "Nikitha Vangale": "Available Weekends 8 AM–11 AM."
}

# --- Node 1: Doctor Selection ---
def get_doctor_details(state: AppointmentState):
    print("\n[INFO] Running doctor selection...")
    doctor_details = doctor_selection_chain.invoke({
        "request": state["request"],   # ✅ correct key
        "context": context
    })
    return Command(goto="human_review", update={"doctor_details": doctor_details})

# --- Node 2: Human Review ---
def human_review(state):
    print("\n[ACTION] Doctor suggested for appointment:")
    print(state.get("doctor_details", "No doctor found."))
    value = interrupt({
        "question": "Are you okay with this doctor? (yes/no)"
    })
    if value.lower().strip() == "yes":
        return Command(goto="schedule_appointment")
    else:
        return Command(goto=END)

# --- Node 3: Schedule Appointment ---
def schedule_appointment(state: AppointmentState):
    print("\n[INFO] Scheduling appointment...")
    doctor_details = state["doctor_details"]
    appointment_details = appointment_schedule_chain.invoke({
        "doctor_details": doctor_details,
        "appointment_context": appointment_context
    })
    return Command(goto=END, update={"appointment_details": appointment_details})


# --- Graph Creation ---
def create_appointment_schedule_workflow():
    graph = StateGraph(AppointmentState)
    graph.add_node("doctor_details", get_doctor_details)
    graph.add_node("human_review", human_review)
    graph.add_node("schedule_appointment", schedule_appointment)

    graph.add_edge(START, "doctor_details")
    return graph.compile(checkpointer=MemorySaver())


# --- Run Workflow ---
appointment_scheduler = create_appointment_schedule_workflow()
input_state = {"request": "I have acne on my face and would like to book an appointment with a doctor."}
thread = {"configurable": {"thread_id": "session_1"}}

result = appointment_scheduler.invoke(input=input_state, config=thread)
print("----------------------input query----------------------")
print("\n user query : ",input_state.get("request"))

print("\n------ Doctor Details ------")
print(result["doctor_details"])

# --- Handle Interrupt ---
tasks = appointment_scheduler.get_state(config=thread).tasks
task = tasks[0]
question = task.interrupts[0].value.get("question")
user_input = input(f"\n{question} ")

# Resume workflow
result = appointment_scheduler.invoke(Command(resume=user_input), config=thread)

print("\n------------ Appointment Details ------------")
print(result.get("appointment_details", "No appointment scheduled."))
