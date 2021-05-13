package com.amazon.opendistroforelasticsearch.sql.opensearch.planner.physical;

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
import java.util.stream.StreamSupport;
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

@RequiredArgsConstructor
@EqualsAndHashCode(callSuper = false)
public class MachineLearningOperator extends PhysicalPlan {
  @Getter
  private final PhysicalPlan input;

  @Getter
  private final String algorithm;

  @Getter
  private final List<MLParameter> parameters;

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
    DataFrame predictionResult = machineLearningClient.predict(algorithm, parameters, inputDataFrame, modelId)
        .actionGet(30, TimeUnit.SECONDS);
    ColumnMeta[] columnMetas = predictionResult.columnMetas();
    iterator = StreamSupport.stream(predictionResult.spliterator(), false)
      .map(row -> convertRowIntoExprValue(columnMetas, row))
      .iterator();
  }

  private ExprValue convertRowIntoExprValue(ColumnMeta[] columnMetas, Row row) {
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

    return ExprTupleValue.fromExprValueMap(resultBuilder.build());
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
