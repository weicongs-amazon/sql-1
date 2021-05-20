package com.amazon.opendistroforelasticsearch.sql.opensearch.planner.physical;

import com.amazon.opendistroforelasticsearch.sql.ast.expression.Argument;
import com.amazon.opendistroforelasticsearch.sql.data.model.ExprDoubleValue;
import com.amazon.opendistroforelasticsearch.sql.data.model.ExprIntegerValue;
import com.amazon.opendistroforelasticsearch.sql.data.model.ExprStringValue;
import com.amazon.opendistroforelasticsearch.sql.data.model.ExprTupleValue;
import com.amazon.opendistroforelasticsearch.sql.data.model.ExprValue;
import com.amazon.opendistroforelasticsearch.sql.planner.physical.PhysicalPlan;
import com.amazon.opendistroforelasticsearch.sql.planner.physical.PhysicalPlanNodeVisitor;
import com.google.common.collect.ImmutableMap;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.opensearch.ml.client.MachineLearningClient;
import org.opensearch.ml.common.dataframe.ColumnMeta;
import org.opensearch.ml.common.dataframe.ColumnValue;
import org.opensearch.ml.common.dataframe.DataFrame;
import org.opensearch.ml.common.dataframe.DataFrameBuilder;
import org.opensearch.ml.common.dataframe.Row;
import org.opensearch.ml.common.parameter.MLParameter;
import org.opensearch.ml.common.parameter.MLParameterBuilder;

/**
 * Machine Learning Physical operator to call machine learning interface to get results for
 * algorithm execution.
 */
@RequiredArgsConstructor
@EqualsAndHashCode(callSuper = false)
public class MachineLearningOperator extends PhysicalPlan {
  @Getter
  private final PhysicalPlan input;

  @Getter
  private final String algorithm;

  @Getter
  private final List<Argument> arguments;

  @Getter
  private final String modelId;

  @Getter
  private final MachineLearningClient machineLearningClient;

  @EqualsAndHashCode.Exclude
  private Iterator<ExprValue> iterator;

  @Override
  public void open() {
    super.open();
    DataFrame inputDataFrame = generateInputDataset();
    List<MLParameter> mlParameters = arguments.stream().map(this::convertArgumentToMLParameter)
        .collect(Collectors.toList());
    DataFrame predictionResult = machineLearningClient
        .predict(algorithm, mlParameters, inputDataFrame, modelId)
        .actionGet(30, TimeUnit.SECONDS);
    Iterator<Row> inputRowIter = inputDataFrame.iterator();
    Iterator<Row> resultRowIter = predictionResult.iterator();
    iterator = new Iterator<ExprValue>() {
      @Override
      public boolean hasNext() {
        return inputRowIter.hasNext();
      }

      @Override
      public ExprValue next() {
        ImmutableMap.Builder<String, ExprValue> resultBuilder = new ImmutableMap.Builder<>();
        resultBuilder.putAll(convertRowIntoExprValue(inputDataFrame.columnMetas(),
            inputRowIter.next()));
        resultBuilder.putAll(convertRowIntoExprValue(predictionResult.columnMetas(),
            resultRowIter.next()));
        return ExprTupleValue.fromExprValueMap(resultBuilder.build());
      }
    };
  }

  protected MLParameter convertArgumentToMLParameter(Argument argument) {
    switch (argument.getValue().getType()) {
      case INTEGER:
        return MLParameterBuilder.parameter(argument.getArgName(),
            (Integer) argument.getValue().getValue());
      case STRING:
        return MLParameterBuilder.parameter(argument.getArgName(),
            (String)argument.getValue().getValue());
      case BOOLEAN:
        return MLParameterBuilder.parameter(argument.getArgName(),
            (Boolean)argument.getValue().getValue());
      case DOUBLE:
        return MLParameterBuilder.parameter(argument.getArgName(),
            (Double)argument.getValue().getValue());
      default:
        throw new IllegalArgumentException("unsupported argument type:"
            + argument.getValue().getType());
    }
  }

  private Map<String, ExprValue> convertRowIntoExprValue(ColumnMeta[] columnMetas, Row row) {
    ImmutableMap.Builder<String, ExprValue> resultBuilder = new ImmutableMap.Builder<>();
    for (int i = 0; i < columnMetas.length; i++) {
      ColumnValue columnValue = row.getValue(i);
      String resultKeyName = columnMetas[i].getName();
      switch (columnValue.columnType()) {
        case INTEGER:
          resultBuilder.put(resultKeyName, new ExprIntegerValue(columnValue.intValue()));
          break;
        case DOUBLE:
          resultBuilder.put(resultKeyName, new ExprDoubleValue(columnValue.doubleValue()));
          break;
        case STRING:
          resultBuilder.put(resultKeyName, new ExprStringValue(columnValue.stringValue()));
          break;
        default:
          break;
      }
    }

    return resultBuilder.build();
  }

  private DataFrame generateInputDataset() {
    List<Map<String, Object>> inputData = new LinkedList<>();
    while (input.hasNext()) {
      Map<String, Object> items = new HashMap<>();
      input.next().tupleValue().forEach((key, value) -> {
        items.put(key, value.value());
      });
      inputData.add(items);
    }

    return DataFrameBuilder.load(inputData);
  }

  @Override
  public <R, C> R accept(PhysicalPlanNodeVisitor<R, C> visitor, C context) {
    return visitor.visitMachineLearning(this, context);
  }

  @Override
  public List<PhysicalPlan> getChild() {
    return Collections.singletonList(input);
  }

  @Override
  public boolean hasNext() {
    return iterator.hasNext();
  }

  @Override
  public ExprValue next() {
    return iterator.next();
  }
}
