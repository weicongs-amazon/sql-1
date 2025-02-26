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

package com.amazon.opendistroforelasticsearch.sql.opensearch.response;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.google.common.collect.ImmutableMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayNameGeneration;
import org.junit.jupiter.api.DisplayNameGenerator;
import org.junit.jupiter.api.Test;

@DisplayNameGeneration(DisplayNameGenerator.ReplaceUnderscores.class)
class OpenSearchAggregationResponseParserTest {

  /**
   * SELECT MAX(age) as max FROM accounts.
   */
  @Test
  void no_bucket_one_metric_should_pass() {
    String response = "{\n"
        + "  \"max#max\": {\n"
        + "    \"value\": 40\n"
        + "  }\n"
        + "}";
    assertThat(parse(response), contains(entry("max", 40d)));
  }

  /**
   * SELECT MAX(age) as max, MIN(age) as min FROM accounts.
   */
  @Test
  void no_bucket_two_metric_should_pass() {
    String response = "{\n"
        + "  \"max#max\": {\n"
        + "    \"value\": 40\n"
        + "  },\n"
        + "  \"min#min\": {\n"
        + "    \"value\": 20\n"
        + "  }\n"
        + "}";
    assertThat(parse(response),
        contains(entry("max", 40d,"min", 20d)));
  }

  @Test
  void one_bucket_one_metric_should_pass() {
    String response = "{\n"
        + "  \"composite#composite_buckets\": {\n"
        + "    \"after_key\": {\n"
        + "      \"type\": \"sale\"\n"
        + "    },\n"
        + "    \"buckets\": [\n"
        + "      {\n"
        + "        \"key\": {\n"
        + "          \"type\": \"cost\"\n"
        + "        },\n"
        + "        \"doc_count\": 2,\n"
        + "        \"avg#avg\": {\n"
        + "          \"value\": 20\n"
        + "        }\n"
        + "      },\n"
        + "      {\n"
        + "        \"key\": {\n"
        + "          \"type\": \"sale\"\n"
        + "        },\n"
        + "        \"doc_count\": 2,\n"
        + "        \"avg#avg\": {\n"
        + "          \"value\": 105\n"
        + "        }\n"
        + "      }\n"
        + "    ]\n"
        + "  }\n"
        + "}";
    assertThat(parse(response),
        containsInAnyOrder(ImmutableMap.of("type", "cost", "avg", 20d),
            ImmutableMap.of("type", "sale", "avg", 105d)));
  }

  @Test
  void two_bucket_one_metric_should_pass() {
    String response = "{\n"
        + "  \"composite#composite_buckets\": {\n"
        + "    \"after_key\": {\n"
        + "      \"type\": \"sale\",\n"
        + "      \"region\": \"us\"\n"
        + "    },\n"
        + "    \"buckets\": [\n"
        + "      {\n"
        + "        \"key\": {\n"
        + "          \"type\": \"cost\",\n"
        + "          \"region\": \"us\"\n"
        + "        },\n"
        + "        \"avg#avg\": {\n"
        + "          \"value\": 20\n"
        + "        }\n"
        + "      },\n"
        + "      {\n"
        + "        \"key\": {\n"
        + "          \"type\": \"sale\",\n"
        + "          \"region\": \"uk\"\n"
        + "        },\n"
        + "        \"avg#avg\": {\n"
        + "          \"value\": 130\n"
        + "        }\n"
        + "      }\n"
        + "    ]\n"
        + "  }\n"
        + "}";
    assertThat(parse(response),
        containsInAnyOrder(ImmutableMap.of("type", "cost", "region", "us", "avg", 20d),
            ImmutableMap.of("type", "sale", "region", "uk", "avg", 130d)));
  }

  @Test
  void unsupported_aggregation_should_fail() {
    String response = "{\n"
        + "  \"date_histogram#max\": {\n"
        + "    \"value\": 40\n"
        + "  }\n"
        + "}";
    IllegalStateException exception =
        assertThrows(IllegalStateException.class, () -> parse(response));
    assertEquals("unsupported aggregation type date_histogram", exception.getMessage());
  }

  @Test
  void nan_value_should_return_null() {
    assertNull(OpenSearchAggregationResponseParser.handleNanValue(Double.NaN));
  }

  /**
   * SELECT AVG(age) FILTER(WHERE age > 37) as filtered FROM accounts.
   */
  @Test
  void filter_aggregation_should_pass() {
    String response = "{\n" 
            +     "    \"filter#filtered\" : {\n" 
            +     "      \"doc_count\" : 3,\n" 
            +     "      \"avg#filtered\" : {\n" 
            +     "        \"value\" : 37.0\n" 
            +     "      }\n" 
            +     "    }\n" 
            +     "  }";
    assertThat(parse(response), contains(entry("filtered", 37.0)));
  }

  /**
   * SELECT AVG(age) FILTER(WHERE age > 37) as filtered FROM accounts GROUP BY gender.
   */
  @Test
  void filter_aggregation_group_by_should_pass() {
    String response = "{\n" 
            + "  \"composite#composite_buckets\":{\n" 
            + "    \"after_key\":{\n" 
            + "      \"gender\":\"m\"\n" 
            + "    },\n" 
            + "    \"buckets\":[\n" 
            + "      {\n" 
            + "        \"key\":{\n" 
            + "          \"gender\":\"f\"\n" 
            + "        },\n" 
            + "        \"doc_count\":3,\n" 
            + "        \"filter#filter\":{\n" 
            + "          \"doc_count\":1,\n" 
            + "          \"avg#avg\":{\n" 
            + "            \"value\":39.0\n" 
            + "          }\n" 
            + "        }\n" 
            + "      },\n" 
            + "      {\n" 
            + "        \"key\":{\n" 
            + "          \"gender\":\"m\"\n" 
            + "        },\n" 
            + "        \"doc_count\":4,\n" 
            + "        \"filter#filter\":{\n" 
            + "          \"doc_count\":2,\n" 
            + "          \"avg#avg\":{\n" 
            + "            \"value\":36.0\n" 
            + "          }\n" 
            + "        }\n" 
            + "      }\n" 
            + "    ]\n" 
            + "  }\n" 
            + "}";
    assertThat(parse(response), containsInAnyOrder(
        entry("gender", "f", "avg", 39.0),
        entry("gender", "m", "avg", 36.0)));
  }

  public List<Map<String, Object>> parse(String json) {
    return OpenSearchAggregationResponseParser.parse(AggregationResponseUtils.fromJson(json));
  }

  public Map<String, Object> entry(String name, Object value) {
    return ImmutableMap.of(name, value);
  }

  public Map<String, Object> entry(String name, Object value, String name2, Object value2) {
    return ImmutableMap.of(name, value, name2, value2);
  }
}