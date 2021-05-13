package com.amazon.opendistroforelasticsearch.sql.opensearch.planner.physical;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Answers.RETURNS_DEEP_STUBS;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.when;

import com.amazon.opendistroforelasticsearch.sql.data.model.ExprIntegerValue;
import com.amazon.opendistroforelasticsearch.sql.data.model.ExprTupleValue;
import com.amazon.opendistroforelasticsearch.sql.data.model.ExprValue;
import com.amazon.opendistroforelasticsearch.sql.planner.physical.PhysicalPlan;
import com.amazon.opendistroforelasticsearch.sql.planner.physical.PhysicalPlanNodeVisitor;
import com.google.common.collect.ImmutableMap;
import java.util.Collections;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.opensearch.ml.client.MachineLearningClient;
import org.opensearch.ml.common.dataframe.DataFrame;
import org.opensearch.ml.common.dataframe.DataFrameBuilder;

@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
public class KmeansOperatorTest {

  @Mock
  private PhysicalPlan input;

  @Mock(answer = RETURNS_DEEP_STUBS)
  private MachineLearningClient machineLearningClient;

  private KmeansOperator kmeansOperator;

  @BeforeEach
  void setup() {
    kmeansOperator = new KmeansOperator(input, 3, machineLearningClient);
    when(input.hasNext()).thenReturn(true).thenReturn(false);
    ImmutableMap.Builder<String, ExprValue> resultBuilder = new ImmutableMap.Builder<>();
    resultBuilder.put("k1", new ExprIntegerValue(2));
    when(input.next()).thenReturn(ExprTupleValue.fromExprValueMap(resultBuilder.build()));

    DataFrame dataFrame = DataFrameBuilder
        .load(Collections.singletonList(ImmutableMap.<String, Object>builder().put("k1", 2D)
            .put("k2", 1)
            .put("k3", "v3")
            .put("k4", true)
            .build())
        );
    when(machineLearningClient.predict(anyString(), anyList(), any())
        .actionGet(anyLong(), eq(TimeUnit.SECONDS)))
        .thenReturn(dataFrame);
  }

  @Test
  public void testOpen() {
    kmeansOperator.open();
    assertTrue(kmeansOperator.hasNext());
    assertNotNull(kmeansOperator.next());
    assertFalse(kmeansOperator.hasNext());
  }

  @Test
  public void testOpen_withNullClusterNo() {
    kmeansOperator = new KmeansOperator(input, null, machineLearningClient);
    kmeansOperator.open();
    assertTrue(kmeansOperator.hasNext());
    assertNotNull(kmeansOperator.next());
    assertFalse(kmeansOperator.hasNext());
  }

  @Test
  public void testAccept() {
    PhysicalPlanNodeVisitor physicalPlanNodeVisitor
        = new PhysicalPlanNodeVisitor<Integer, Object>() {};
    assertNull(kmeansOperator.accept(physicalPlanNodeVisitor, null));
  }
}