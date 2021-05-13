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
 *
 *    Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License").
 *    You may not use this file except in compliance with the License.
 *    A copy of the License is located at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    or in the "license" file accompanying this file. This file is distributed
 *    on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *    express or implied. See the License for the specific language governing
 *    permissions and limitations under the License.
 *
 */

package com.amazon.opendistroforelasticsearch.sql.opensearch.storage;

import static com.amazon.opendistroforelasticsearch.sql.planner.logical.LogicalPlanDSL.relation;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.amazon.opendistroforelasticsearch.sql.opensearch.client.OpenSearchClient;
import com.amazon.opendistroforelasticsearch.sql.planner.logical.LogicalKmeans;
import com.amazon.opendistroforelasticsearch.sql.planner.logical.LogicalPlan;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Answers;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
public class OpenSearchDefaultImplementorTest {

  @Mock
  OpenSearchIndexScan indexScan;

  @Mock
  OpenSearchClient client;

  /**
   * For test coverage.
   */
  @Test
  public void visitInvalidTypeShouldThrowException() {
    final OpenSearchIndex.OpenSearchDefaultImplementor implementor =
        new OpenSearchIndex.OpenSearchDefaultImplementor(indexScan, client);

    final IllegalStateException exception =
        assertThrows(IllegalStateException.class, () -> implementor.visitNode(relation("index"),
            indexScan));
    ;
    assertEquals(
        "unexpected plan node type "
            + "class com.amazon.opendistroforelasticsearch.sql.planner.logical.LogicalRelation",
        exception.getMessage());
  }

  @Test
  public void visitKmeans() {
    LogicalKmeans node = Mockito.mock(LogicalKmeans.class, Answers.RETURNS_DEEP_STUBS);
    Mockito.when(node.getChild().get(0)).thenReturn(Mockito.mock(LogicalPlan.class));
    OpenSearchIndex.OpenSearchDefaultImplementor implementor =
        new OpenSearchIndex.OpenSearchDefaultImplementor(indexScan, client);
    assertNotNull(implementor.visitKmeans(node, indexScan));
  }
}
