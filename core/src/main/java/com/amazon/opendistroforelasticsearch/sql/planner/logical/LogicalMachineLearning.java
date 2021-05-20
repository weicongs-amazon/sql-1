package com.amazon.opendistroforelasticsearch.sql.planner.logical;

import com.amazon.opendistroforelasticsearch.sql.ast.expression.Argument;
import java.util.Collections;
import java.util.List;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.ToString;

/**
 * Machine Learning Logical Plan.
 */
@Getter
@ToString
@EqualsAndHashCode(callSuper = true)
public class LogicalMachineLearning extends LogicalPlan {

  private final String algorithm;

  private final List<Argument> arguments;

  private final String modelId;

  /**
   * Constructor of LogicalMachineLearning.
   * @param child child logical plan
   * @param algorithm algorithm name
   * @param arguments arguments of algorithm
   */
  public LogicalMachineLearning(LogicalPlan child,
                                String algorithm,
                                List<Argument> arguments) {
    this(child, algorithm, arguments, null);
  }

  /**
   * Constructor of LogicalMachineLearning.
   * @param child child logical plan
   * @param algorithm algorithm name
   * @param arguments arguments of algorithm
   * @param modelId id of model
   */
  public LogicalMachineLearning(LogicalPlan child,
                                String algorithm,
                                List<Argument> arguments,
                                String modelId) {
    super(Collections.singletonList(child));
    this.algorithm = algorithm;
    this.arguments = arguments;
    this.modelId = modelId;
  }

  @Override
  public <R, C> R accept(LogicalPlanNodeVisitor<R, C> visitor, C context) {
    return visitor.visitMachineLearning(this, context);
  }
}
