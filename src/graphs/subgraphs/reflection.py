"""Reflection Subgraph for LangGraph workflows.

Simple subgraph that handles summary quality evaluation and improvement
using the existing reflection utilities.
"""

from langgraph.graph import StateGraph, START, END

from src.utils.graph_schemas import ReflectionState
from src.utils.reflection_utils import apply_reflection_to_summary


def build_reflection_subgraph() -> StateGraph:
    """Build reflection subgraph for summary improvement.
    
    **Pipeline:**
    1. Apply reflection to summary if enabled
    2. Return improved summary with metadata
    
    Returns:
        Compiled StateGraph for reflection processing
    """
    
    def apply_reflection(state: ReflectionState) -> ReflectionState:
        """Apply reflection to improve summary quality.
        
        Uses existing reflection utilities with simplified interface.
        """
        summary_text = state["summary_text"]
        topic = state["topic"]
        length_requirement = state["length_requirement"]
        source_content = state["source_content"]
        enable_reflection = state["enable_reflection"]
        
        if not enable_reflection or not summary_text:
            print(f"⏭️ Skipping reflection for topic '{topic}'")
            return {
                **state,
                "final_summary": summary_text,
                "reflection_applied": False,
                "reflection_metadata": {"skipped": True, "reason": "disabled or empty summary"}
            }
        
        print(f"🔍 Applying reflection to summary for topic '{topic}'")
        
        try:
            # Use existing reflection logic
            reflection_result = apply_reflection_to_summary(
                summary_text=summary_text,
                topic=topic,
                length_requirement=length_requirement,
                source_content=source_content
            )
            
            # Extract improved summary
            improved_summary = reflection_result.get("improved_summary")
            
            if improved_summary and hasattr(improved_summary, 'improved_text') and improved_summary.improved_text:
                final_text = improved_summary.improved_text
                reflection_applied = True
                metadata = {
                    "reflection_applied": True,
                    "changes_made": getattr(improved_summary, 'changes_made', []),
                    "initial_evaluation": reflection_result.get("evaluation").__dict__ if reflection_result.get("evaluation") else None,
                    "final_evaluation": getattr(improved_summary, 'final_evaluation', {})
                }
                print(f"✨ Reflection completed for topic '{topic}'")
            else:
                final_text = summary_text
                reflection_applied = False
                metadata = {
                    "reflection_applied": False,
                    "error": reflection_result.get("error", "Unknown reflection error")
                }
                print(f"⚠️ Reflection failed for topic '{topic}', using original summary")
            
            return {
                **state,
                "final_summary": final_text,
                "reflection_applied": reflection_applied,
                "reflection_metadata": metadata
            }
            
        except Exception as e:
            print(f"❌ Reflection error for topic '{topic}': {str(e)}")
            return {
                **state,
                "final_summary": summary_text,
                "reflection_applied": False,
                "reflection_metadata": {
                    "reflection_applied": False,
                    "error": f"Reflection failed: {str(e)}"
                }
            }
    
    # Build the simple subgraph
    graph = StateGraph(ReflectionState)
    
    # Add single node for reflection
    graph.add_node("apply_reflection", apply_reflection)
    
    # Simple linear flow
    graph.add_edge(START, "apply_reflection")
    graph.add_edge("apply_reflection", END)
    
    return graph.compile()


# Create the compiled subgraph instance
REFLECTION_SUBGRAPH = build_reflection_subgraph()