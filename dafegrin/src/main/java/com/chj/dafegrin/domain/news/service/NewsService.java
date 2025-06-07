package com.chj.dafegrin.domain.news.service;
import com.chj.dafegrin.entity.News;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.jsoup.Jsoup;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.*;
import java.net.*;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.*;

@Service
@Transactional
@Slf4j
@RequiredArgsConstructor
public class NewsService {
    @Value("${naver.client-id}")
    private String CLIENT_ID;

    @Value("${naver.client-secret}")
    private String CLIENT_SECRET;

    @Value("${naver.news-search-api}")
    private String NEWS_SEARCH_API;

    public List<News> getNews(String keyword) {
        String responseBody = getResponseBody(keyword);
        return parse(responseBody);
    }

    /**
     * NaverNewsApi를 통해 받은 JSON 값을 파싱하여 제목 + 내용(일부)로 결합하여 List로 반환
     */
    private String getResponseBody(String keyword) {
        String text = null;
        try {
            text = URLEncoder.encode(keyword, "UTF-8");
        } catch (UnsupportedEncodingException e) {
            throw new RuntimeException("검색어 인코딩 실패",e);
        }
        String apiURL = NEWS_SEARCH_API + text;    // JSON 결과

        Map<String, String> requestHeaders = new HashMap<>();
        requestHeaders.put("X-Naver-Client-Id", CLIENT_ID);
        requestHeaders.put("X-Naver-Client-Secret", CLIENT_SECRET);

        return get(apiURL,requestHeaders);
    }

    private String get(String apiUrl, Map<String, String> requestHeaders){
        HttpURLConnection con = connect(apiUrl);
        try {
            con.setRequestMethod("GET");
            for(Map.Entry<String, String> header :requestHeaders.entrySet()) {
                con.setRequestProperty(header.getKey(), header.getValue());
            }

            int responseCode = con.getResponseCode();
            if (responseCode == HttpURLConnection.HTTP_OK) { // 정상 호출
                return readBody(con.getInputStream());
            } else { // 오류 발생
                return readBody(con.getErrorStream());
            }
        } catch (IOException e) {
            throw new RuntimeException("API 요청과 응답 실패", e);
        } finally {
            con.disconnect();
        }
    }

    private HttpURLConnection connect(String apiUrl){
        try {
            URL url = (new URI(apiUrl)).toURL();
            return (HttpURLConnection)url.openConnection();
        } catch (IOException | URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

    private String readBody(InputStream body){
        InputStreamReader streamReader = new InputStreamReader(body);

        try (BufferedReader lineReader = new BufferedReader(streamReader)) {
            StringBuilder responseBody = new StringBuilder();

            String line;
            while ((line = lineReader.readLine()) != null) {
                responseBody.append(line);
            }

            return responseBody.toString();
        } catch (IOException e) {
            throw new RuntimeException("API 응답을 읽는 데 실패했습니다.", e);
        }
    }

    /**
     * 뉴스 객체 리스트로 변경 + 뉴스 내용만 따로 담아주는 List<String> 메서드 작성
     */
    private List<News> parse(String resBody) {
        List<News> result = new ArrayList<>();

        try {
            ObjectMapper objectMapper = new ObjectMapper();
            JsonNode rootNode = objectMapper.readTree(resBody);
            JsonNode itemNode = rootNode.path("items");
            for(JsonNode item : itemNode) {
                String pubDate = item.path("pubDate").asText();
                LocalDateTime date = toLocalDateTime(pubDate);
                if(isBeforeYesterday6PM(date)) continue;

                String title = item.path("title").asText();
                title = Jsoup.parse(title).text();
                String description = item.path("description").asText();
                description = Jsoup.parse(description).text();

                String content = title + ". " + description;

                News news = new News();
                news.setDate(date);
                news.setContent(content);
                result.add(news);
            }
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
        return result;
    }

    private LocalDateTime toLocalDateTime(String pubDate) {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("EEE, dd MMM yyyy HH:mm:ss Z", Locale.ENGLISH);
        return LocalDateTime.parse(pubDate, formatter);
    }

    private boolean isBeforeYesterday6PM(LocalDateTime date) {
        LocalDateTime now = LocalDateTime.now(ZoneId.of("Asia/Seoul"));
        LocalDateTime yesterday6PM = now.minusDays(1)
                .withHour(18)
                .withMinute(0)
                .withMinute(0)
                .withNano(0);

        return date.isBefore(yesterday6PM);
    }
}
