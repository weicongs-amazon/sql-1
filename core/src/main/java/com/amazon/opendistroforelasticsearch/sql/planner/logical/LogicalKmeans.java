package com.amazon.opendistroforelasticsearch.sql.planner.logical;

import java.util.Collections;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.ToString;

@Getter
@ToString
@EqualsAndHashCode(callSuper = true)
public class LogicalKmeans extends LogicalPlan {

  private final Integer numberOfClusters;

  public LogicalKmeans(LogicalPlan child, Integer numberOfClusters) {
    super(Collections.singletonList(child));
    this.numberOfClusters = numberOfClusters;
  }

  @Override
  public <R, C> R accept(LogicalPlanNodeVisitor<R, C> visitor, C context) {
    return visitor.visitKmeans(this, context);
  }
}
