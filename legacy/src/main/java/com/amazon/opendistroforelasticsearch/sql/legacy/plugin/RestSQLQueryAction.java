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

package com.amazon.opendistroforelasticsearch.sql.legacy.plugin;

import static com.amazon.opendistroforelasticsearch.sql.executor.ExecutionEngine.QueryResponse;
import static com.amazon.opendistroforelasticsearch.sql.protocol.response.format.JsonResponseFormatter.Style.PRETTY;
import static org.opensearch.rest.RestStatus.INTERNAL_SERVER_ERROR;
import static org.opensearch.rest.RestStatus.OK;

import com.amazon.opendistroforelasticsearch.sql.common.antlr.SyntaxCheckException;
import com.amazon.opendistroforelasticsearch.sql.common.response.ResponseListener;
import com.amazon.opendistroforelasticsearch.sql.common.setting.Settings;
import com.amazon.opendistroforelasticsearch.sql.executor.ExecutionEngine.ExplainResponse;
import com.amazon.opendistroforelasticsearch.sql.legacy.metrics.MetricName;
import com.amazon.opendistroforelasticsearch.sql.legacy.metrics.Metrics;
import com.amazon.opendistroforelasticsearch.sql.opensearch.security.SecurityAccess;
import com.amazon.opendistroforelasticsearch.sql.planner.physical.PhysicalPlan;
import com.amazon.opendistroforelasticsearch.sql.protocol.response.QueryResult;
import com.amazon.opendistroforelasticsearch.sql.protocol.response.format.CsvResponseFormatter;
import com.amazon.opendistroforelasticsearch.sql.protocol.response.format.Format;
import com.amazon.opendistroforelasticsearch.sql.protocol.response.format.JdbcResponseFormatter;
import com.amazon.opendistroforelasticsearch.sql.protocol.response.format.JsonResponseFormatter;
import com.amazon.opendistroforelasticsearch.sql.protocol.response.format.RawResponseFormatter;
import com.amazon.opendistroforelasticsearch.sql.protocol.response.format.ResponseFormatter;
import com.amazon.opendistroforelasticsearch.sql.sql.SQLService;
import com.amazon.opendistroforelasticsearch.sql.sql.config.SQLServiceConfig;
import com.amazon.opendistroforelasticsearch.sql.sql.domain.SQLQueryRequest;
import java.io.IOException;
import java.security.PrivilegedExceptionAction;
import java.util.List;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.client.node.NodeClient;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.BytesRestResponse;
import org.opensearch.rest.RestChannel;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.RestStatus;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;

/**
 * New SQL REST action handler. This will not be registered to OpenSearch unless:
 *  1) we want to test new SQL engine;
 *  2) all old functionalities migrated to new query engine and legacy REST handler removed.
 */
public class RestSQLQueryAction extends BaseRestHandler {

  private static final Logger LOG = LogManager.getLogger();

  public static final RestChannelConsumer NOT_SUPPORTED_YET = null;

  private final ClusterService clusterService;

  /**
   * Settings required by been initialization.
   */
  private final Settings pluginSettings;

  /**
   * Constructor of RestSQLQueryAction.
   */
  public RestSQLQueryAction(ClusterService clusterService, Settings pluginSettings) {
    super();
    this.clusterService = clusterService;
    this.pluginSettings = pluginSettings;
  }

  @Override
  public String getName() {
    return "sql_query_action";
  }

  @Override
  public List<Route> routes() {
    throw new UnsupportedOperationException("New SQL handler is not ready yet");
  }

  @Override
  protected RestChannelConsumer prepareRequest(RestRequest request, NodeClient nodeClient) {
    throw new UnsupportedOperationException("New SQL handler is not ready yet");
  }

  /**
   * Prepare REST channel consumer for a SQL query request.
   * @param request     SQL request
   * @param nodeClient  node client
   * @return            channel consumer
   */
  public RestChannelConsumer prepareRequest(SQLQueryRequest request, NodeClient nodeClient) {
    if (!request.isSupported()) {
      return NOT_SUPPORTED_YET;
    }

    SQLService sqlService = createSQLService(nodeClient);
    PhysicalPlan plan;
    try {
      // For now analyzing and planning stage may throw syntax exception as well
      // which hints the fallback to legacy code is necessary here.
      plan = sqlService.plan(
                sqlService.analyze(
                    sqlService.parse(request.getQuery())));
    } catch (SyntaxCheckException e) {
      // When explain, print info log for what unsupported syntax is causing fallback to old engine
      if (request.isExplainRequest()) {
        LOG.info("Request is falling back to old SQL engine due to: " + e.getMessage());
      }
      return NOT_SUPPORTED_YET;
    }

    if (request.isExplainRequest()) {
      return channel -> sqlService.explain(plan, createExplainResponseListener(channel));
    }
    return channel -> sqlService.execute(plan, createQueryResponseListener(channel, request));
  }

  private SQLService createSQLService(NodeClient client) {
    return doPrivileged(() -> {
      AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext();
      context.registerBean(ClusterService.class, () -> clusterService);
      context.registerBean(NodeClient.class, () -> client);
      context.registerBean(Settings.class, () -> pluginSettings);
      context.register(OpenSearchSQLPluginConfig.class);
      context.register(SQLServiceConfig.class);
      context.refresh();
      return context.getBean(SQLService.class);
    });
  }

  private ResponseListener<ExplainResponse> createExplainResponseListener(RestChannel channel) {
    return new ResponseListener<ExplainResponse>() {
      @Override
      public void onResponse(ExplainResponse response) {
        sendResponse(channel, OK, new JsonResponseFormatter<ExplainResponse>(PRETTY) {
          @Override
          protected Object buildJsonObject(ExplainResponse response) {
            return response;
          }
        }.format(response));
      }

      @Override
      public void onFailure(Exception e) {
        LOG.error("Error happened during explain", e);
        logAndPublishMetrics(e);
        sendResponse(channel, INTERNAL_SERVER_ERROR,
            "Failed to explain the query due to error: " + e.getMessage());
      }
    };
  }

  private ResponseListener<QueryResponse> createQueryResponseListener(RestChannel channel, SQLQueryRequest request) {
    Format format = request.format();
    ResponseFormatter<QueryResult> formatter;
    if (format.equals(Format.CSV)) {
      formatter = new CsvResponseFormatter(request.sanitize());
    } else if (format.equals(Format.RAW)) {
      formatter = new RawResponseFormatter();
    } else {
      formatter = new JdbcResponseFormatter(PRETTY);
    }
    return new ResponseListener<QueryResponse>() {
      @Override
      public void onResponse(QueryResponse response) {
        sendResponse(channel, OK,
            formatter.format(new QueryResult(response.getSchema(), response.getResults())));
      }

      @Override
      public void onFailure(Exception e) {
        LOG.error("Error happened during query handling", e);
        logAndPublishMetrics(e);
        sendResponse(channel, INTERNAL_SERVER_ERROR, formatter.format(e));
      }
    };
  }

  private <T> T doPrivileged(PrivilegedExceptionAction<T> action) {
    try {
      return SecurityAccess.doPrivileged(action);
    } catch (IOException e) {
      throw new IllegalStateException("Failed to perform privileged action", e);
    }
  }

  private void sendResponse(RestChannel channel, RestStatus status, String content) {
    channel.sendResponse(new BytesRestResponse(
        status, "application/json; charset=UTF-8", content));
  }

  private static void logAndPublishMetrics(Exception e) {
    LOG.error("Server side error during query execution", e);
    Metrics.getInstance().getNumericalMetric(MetricName.FAILED_REQ_COUNT_SYS).increment();
  }
}
