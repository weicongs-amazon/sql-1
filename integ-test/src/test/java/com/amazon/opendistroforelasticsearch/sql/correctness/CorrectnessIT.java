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
 *   Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License").
 *   You may not use this file except in compliance with the License.
 *   A copy of the License is located at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   or in the "license" file accompanying this file. This file is distributed
 *   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *   express or implied. See the License for the specific language governing
 *   permissions and limitations under the License.
 */

package com.amazon.opendistroforelasticsearch.sql.correctness;

import static com.amazon.opendistroforelasticsearch.sql.util.TestUtils.getResourceFilePath;

import com.amazon.opendistroforelasticsearch.sql.correctness.report.TestReport;
import com.amazon.opendistroforelasticsearch.sql.correctness.runner.ComparisonTest;
import com.amazon.opendistroforelasticsearch.sql.correctness.runner.connection.DBConnection;
import com.amazon.opendistroforelasticsearch.sql.correctness.runner.connection.OpenSearchConnection;
import com.amazon.opendistroforelasticsearch.sql.correctness.runner.connection.JDBCConnection;
import com.amazon.opendistroforelasticsearch.sql.correctness.testset.TestDataSet;
import com.amazon.opendistroforelasticsearch.sql.legacy.CustomExternalTestCluster;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import com.google.common.collect.Maps;
import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Map;
import java.util.TimeZone;
import org.apache.http.HttpHost;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.json.JSONObject;
import org.junit.Test;
import org.opensearch.client.RestClient;
import org.opensearch.common.Strings;
import org.opensearch.common.transport.TransportAddress;
import org.opensearch.test.OpenSearchIntegTestCase;
import org.opensearch.test.TestCluster;

/**
 * Correctness integration test by performing comparison test with other databases.
 */
@OpenSearchIntegTestCase.SuiteScopeTestCase
@OpenSearchIntegTestCase.ClusterScope(scope = OpenSearchIntegTestCase.Scope.SUITE, numDataNodes = 3, supportsDedicatedMasters = false, transportClientRatio = 1)
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class CorrectnessIT extends OpenSearchIntegTestCase {

  private static final Logger LOG = LogManager.getLogger();

  @Test
  public void performComparisonTest() {
    TestConfig config = new TestConfig(getCmdLineArgs());
    LOG.info("Starting comparison test {}", config);

    try (ComparisonTest test = new ComparisonTest(getThisDBConnection(config),
        getOtherDBConnections(config))) {
      LOG.info("Loading test data set...");
      test.connect();
      for (TestDataSet dataSet : config.getTestDataSets()) {
        test.loadData(dataSet);
      }

      LOG.info("Verifying test queries...");
      TestReport report = test.verify(config.getTestQuerySet());

      LOG.info("Saving test report to disk...");
      store(report);

      LOG.info("Cleaning up test data...");
      for (TestDataSet dataSet : config.getTestDataSets()) {
        test.cleanUp(dataSet);
      }
    }
    LOG.info("Completed comparison test.");
  }

  private Map<String, String> getCmdLineArgs() {
    return Maps.fromProperties(System.getProperties());
  }

  private DBConnection getThisDBConnection(TestConfig config) {
    String dbUrl = config.getDbConnectionUrl();
    if (dbUrl.isEmpty()) {
      return getOpenSearchConnection(config);
    }
    return new JDBCConnection("DB Tested", dbUrl);
  }

  /**
   * Use OpenSearch cluster given on CLI arg or internal embedded in SQLIntegTestCase
   */
  private DBConnection getOpenSearchConnection(TestConfig config) {
    RestClient client;
    String openSearchHost = config.getOpenSearchHostUrl();
    if (openSearchHost.isEmpty()) {
      client = getRestClient();
      openSearchHost = client.getNodes().get(0).getHost().toString();
    } else {
      client = RestClient.builder(HttpHost.create(openSearchHost)).build();
    }
    return new OpenSearchConnection("jdbc:opensearch://" + openSearchHost, client);
  }

  /**
   * Create database connection with database name and connect URL
   */
  private DBConnection[] getOtherDBConnections(TestConfig config) {
    return config.getOtherDbConnectionNameAndUrls().
        entrySet().stream().
        map(e -> new JDBCConnection(e.getKey(), e.getValue())).
        toArray(DBConnection[]::new);
  }

  private void store(TestReport report) {
    try {
      // Create reports folder if not exists
      String folderPath = "reports/";
      Path path = Paths.get(getResourceFilePath(folderPath));
      if (Files.notExists(path)) {
        Files.createDirectory(path);
      }

      // Write to report file
      String relFilePath = folderPath + reportFileName();
      String absFilePath = getResourceFilePath(relFilePath);
      byte[] content = new JSONObject(report).toString(2).getBytes();

      LOG.info("Report file location is {}", absFilePath);
      Files.write(Paths.get(absFilePath), content);
    } catch (Exception e) {
      throw new IllegalStateException("Failed to store report file", e);
    }
  }

  private String reportFileName() {
    SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd-HH");
    df.setTimeZone(TimeZone.getTimeZone("GMT"));
    String dateTime = df.format(new Date());
    return "report_" + dateTime + ".json";
  }

  @Override
  protected TestCluster buildTestCluster(Scope scope, long seed) throws IOException {

    String clusterAddresses = System.getProperty(TESTS_CLUSTER);

    if (Strings.hasLength(clusterAddresses)) {
      String[] stringAddresses = clusterAddresses.split(",");
      TransportAddress[] transportAddresses = new TransportAddress[stringAddresses.length];
      int i = 0;
      for (String stringAddress : stringAddresses) {
        URL url = new URL("http://" + stringAddress);
        InetAddress inetAddress = InetAddress.getByName(url.getHost());
        transportAddresses[i++] =
            new TransportAddress(new InetSocketAddress(inetAddress, url.getPort()));
      }
      return new CustomExternalTestCluster(createTempDir(), externalClusterClientSettings(),
          transportClientPlugins(), transportAddresses);
    }
    return super.buildTestCluster(scope, seed);
  }

}
