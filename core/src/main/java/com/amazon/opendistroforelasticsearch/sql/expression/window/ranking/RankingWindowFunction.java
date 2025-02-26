/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

/*
 *    Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License").
 *    You may not use this file except in compliance with the License.
 *    A copy of the License is located at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    or in the "license" file accompanying this file. This file is distributed
 *    on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *    express or implied. See the License for the specific language governing
 *    permissions and limitations under the License.
 *
 */

package com.amazon.opendistroforelasticsearch.sql.expression.window.ranking;

import static java.util.Collections.emptyList;

import com.amazon.opendistroforelasticsearch.sql.data.model.ExprIntegerValue;
import com.amazon.opendistroforelasticsearch.sql.data.model.ExprValue;
import com.amazon.opendistroforelasticsearch.sql.data.type.ExprCoreType;
import com.amazon.opendistroforelasticsearch.sql.data.type.ExprType;
import com.amazon.opendistroforelasticsearch.sql.expression.Expression;
import com.amazon.opendistroforelasticsearch.sql.expression.FunctionExpression;
import com.amazon.opendistroforelasticsearch.sql.expression.env.Environment;
import com.amazon.opendistroforelasticsearch.sql.expression.function.FunctionName;
import com.amazon.opendistroforelasticsearch.sql.expression.window.WindowDefinition;
import com.amazon.opendistroforelasticsearch.sql.expression.window.WindowFunctionExpression;
import com.amazon.opendistroforelasticsearch.sql.expression.window.frame.CurrentRowWindowFrame;
import com.amazon.opendistroforelasticsearch.sql.expression.window.frame.WindowFrame;
import com.amazon.opendistroforelasticsearch.sql.storage.bindingtuple.BindingTuple;
import java.util.List;
import java.util.stream.Collectors;
import org.apache.commons.lang3.tuple.Pair;

/**
 * Ranking window function base class that captures same info across different ranking functions,
 * such as same return type (integer), same argument list (no arg).
 */
public abstract class RankingWindowFunction extends FunctionExpression
                                            implements WindowFunctionExpression {

  /**
   * Current rank number assigned.
   */
  protected int rank;

  public RankingWindowFunction(FunctionName functionName) {
    super(functionName, emptyList());
  }

  @Override
  public ExprType type() {
    return ExprCoreType.INTEGER;
  }

  @Override
  public WindowFrame createWindowFrame(WindowDefinition definition) {
    return new CurrentRowWindowFrame(definition);
  }

  @Override
  public ExprValue valueOf(Environment<Expression, ExprValue> valueEnv) {
    return new ExprIntegerValue(rank((CurrentRowWindowFrame) valueEnv));
  }

  /**
   * Rank logic that sub-class needs to implement.
   * @param frame   window frame
   * @return        rank number
   */
  protected abstract int rank(CurrentRowWindowFrame frame);

  /**
   * Check sort field to see if current value is different from previous.
   * @param frame   window frame
   * @return        true if different, false if same or no sort list defined
   */
  protected boolean isSortFieldValueDifferent(CurrentRowWindowFrame frame) {
    if (isSortItemsNotDefined(frame)) {
      return false;
    }

    List<Expression> sortItems = frame.getWindowDefinition()
                                      .getSortList()
                                      .stream()
                                      .map(Pair::getRight)
                                      .collect(Collectors.toList());

    List<ExprValue> previous = resolve(frame, sortItems, frame.previous());
    List<ExprValue> current = resolve(frame, sortItems, frame.current());
    return !current.equals(previous);
  }

  private boolean isSortItemsNotDefined(CurrentRowWindowFrame frame) {
    return frame.getWindowDefinition().getSortList().isEmpty();
  }

  private List<ExprValue> resolve(WindowFrame frame, List<Expression> expressions, ExprValue row) {
    BindingTuple valueEnv = row.bindingTuples();
    return expressions.stream()
                      .map(expr -> expr.valueOf(valueEnv))
                      .collect(Collectors.toList());
  }

  @Override
  public String toString() {
    return getFunctionName() + "()";
  }
}
